#!/usr/bin/env python3
"""Which activation tensors are alive at each fused layer, and a replay of ONNX Runtime's arena on them.

Lifetimes (same rule as layer_stats.py and ONNX Runtime's allocation planner): a tensor is alive from the
fused layer that writes it until the last fused layer that reads it; while a layer runs, its inputs and its
outputs are alive together. The graph is fixed, so the lifetimes are the same in every run.

Arena replay: the run-1 allocations (outputs allocated when their layer starts, tensors freed after their
last reader) are played through a simple model of ONNX Runtime's BFC arena with its default settings, as
logged by arena_log_check.py:
  - starts with a 1 MiB region; a request that fits in no free piece takes a new region from the OS, whose
    size starts at 1 MiB and doubles until it covers the request (kNextPowerOfTwo); after an extension that
    did not need doubling, the next region size doubles anyway
  - best fit: the smallest free piece that is large enough
  - a free piece is split only if it is at least twice the request or the waste would be >= 128 MiB
    (max_dead_bytes_per_chunk); otherwise the whole piece is used
  - freed neighbouring pieces in the same region merge; pieces in different regions do not
  - pages written = the bytes of each tensor placed in a region (page size from the OS): an estimate of RSS
Run 2 is replayed the same way (tensor by tensor, as with the memory pattern off). Not modelled: kernel
scratch buffers (also taken from the arena), ONNX Runtime's planner reusing a dead buffer of the same size,
and the planned block of the memory pattern.

Inputs (run folder of profile_layers.py): optimized_model.onnx, layer_stats.csv; original model for shapes;
optionally the latest results/arena_log_check/run_*/segments.csv for comparison.
Outputs (in the run folder): tensor_lifetimes.csv, arena_replay.json, plots/tensor_lifetimes.png

Usage:
  python tensor_lifetimes.py
  python tensor_lifetimes.py --run-dir results/exp_a_profile/run_20261015_115052
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnx
from matplotlib.patches import Patch
from onnx import shape_inference

from analyze_profile import BLUE, ORANGE, INK, MUTED, HERE, read_csv, style, write_csv
from layer_stats import nbytes, tensor_info
from plot_layer_memory import block_spans

MIB = 2 ** 20
PAGE = os.sysconf("SC_PAGE_SIZE")
GREY = "#A9A396"


def lifetimes(run_dir, original_path):
    """One row per activation tensor: writer ID, last reader ID, MiB, kind; plus the layer rows."""
    shapes = tensor_info(shape_inference.infer_shapes(onnx.load(original_path)))
    opt = onnx.load(os.path.join(run_dir, "optimized_model.onnx"))
    weights = {i.name for i in opt.graph.initializer}
    layers = sorted(read_csv(os.path.join(run_dir, "layer_stats.csv")), key=lambda r: int(r["order"]))
    nodes = {n.name: n for n in opt.graph.node}
    graph_inputs = {i.name for i in opt.graph.input} - weights
    graph_outputs = {o.name for o in opt.graph.output}

    first, last, writer_op = {t: 0 for t in graph_inputs}, {}, {}
    for idx, r in enumerate(layers):
        n = nodes[r["layer"]]
        for t in n.input:
            if t and t not in weights:
                last[t] = idx
        for t in n.output:
            first[t], writer_op[t] = idx, r["op_type"]
            last.setdefault(t, idx)
    for t in graph_outputs:
        last[t] = len(layers) - 1

    rows = []
    for t in sorted(first, key=lambda t: (first[t], t not in graph_inputs)):
        kind = ("model input" if t in graph_inputs else "model output" if t in graph_outputs
                else "kept for later" if last[t] - first[t] > 1 else "next layer only")
        rows.append({"tensor": t, "written_by": "input" if t in graph_inputs else first[t],
                     "writer_op": writer_op.get(t, ""), "last_read_by": last[t],
                     "MiB": round(nbytes(shapes, t) / MIB, 3), "bytes": nbytes(shapes, t), "kind": kind})
    return rows, layers


class Arena:
    """Simple model of ONNX Runtime's BFCArena (default CPU settings, see module docstring)."""

    def __init__(self, initial=MIB, max_dead=128 * MIB):
        self.max_dead, self.next_size = max_dead, initial
        self.regions, self.chunks, self.extensions = [], [], []
        self.pages = defaultdict(set)                     # region -> pages written so far
        self._new_region(initial)

    def _new_region(self, size):
        self.regions.append(size)
        self.chunks.append({"region": len(self.regions) - 1, "off": 0, "size": size, "free": True})

    def _extend(self, nb):
        doubled = False
        while nb > self.next_size:
            self.next_size *= 2
            doubled = True
        size = self.next_size
        self._new_region(size)
        if not doubled:
            self.next_size *= 2
        return size

    def alloc(self, nb_exact, tag):
        nb = -(-nb_exact // 256) * 256
        fits = [c for c in self.chunks if c["free"] and c["size"] >= nb]
        if not fits:
            self.extensions.append({**tag, "requested_MiB": round(nb_exact / MIB, 3),
                                    "region_MiB": round(self._extend(nb) / MIB, 3)})
            fits = [self.chunks[-1]]
        c = min(fits, key=lambda c: (c["size"], c["region"], c["off"]))
        if c["size"] >= 2 * nb or c["size"] - nb >= self.max_dead:
            rest = {"region": c["region"], "off": c["off"] + nb, "size": c["size"] - nb, "free": True}
            c["size"] = nb
            self.chunks.insert(self.chunks.index(c) + 1, rest)
        c["free"] = False
        self.pages[c["region"]].update(range(c["off"] // PAGE, (c["off"] + nb_exact - 1) // PAGE + 1))
        return c

    def free(self, c):
        c["free"] = True
        merged = []
        for ch in sorted(self.chunks, key=lambda ch: (ch["region"], ch["off"])):
            if merged and merged[-1]["region"] == ch["region"] and merged[-1]["free"] and ch["free"]:
                merged[-1]["size"] += ch["size"]
            else:
                merged.append(ch)
        self.chunks = merged

    def reserved_mib(self):
        return sum(self.regions) / MIB

    def written_mib(self):
        return sum(len(p) for p in self.pages.values()) * PAGE / MIB


def replay(tensors, n_layers, runs=2):
    """Allocate outputs when their layer starts, free after the last reader; graph outputs at the end."""
    arena = Arena()
    out_of, ends_at = defaultdict(list), defaultdict(list)
    for t in tensors:
        if t["kind"] == "model input":                   # owned by Python (NumPy), not the arena
            continue
        out_of[t["written_by"]].append(t)
        if t["kind"] != "model output":
            ends_at[t["last_read_by"]].append(t)
    per_run = []
    for run in range(1, runs + 1):
        ext_before, written_before = len(arena.extensions), arena.written_mib()
        live = {}
        for idx in range(n_layers):
            for t in out_of[idx]:
                live[t["tensor"]] = arena.alloc(t["bytes"], {"run": run, "layer_id": idx, "tensor_MiB": t["MiB"]})
            for t in ends_at[idx]:
                arena.free(live.pop(t["tensor"]))
        for name in list(live):                           # model output returned to Python, then freed
            arena.free(live.pop(name))
        per_run.append({"run": run, "new_regions": arena.extensions[ext_before:],
                        "reserved_MiB": round(arena.reserved_mib(), 2),
                        "newly_written_MiB": round(arena.written_mib() - written_before, 2),
                        "written_total_MiB": round(arena.written_mib(), 2)})
    return per_run


def plot(tensors, layers, extensions, out):
    n = len(layers)
    spans = block_spans([r["group"] for r in layers])
    color = {"next layer only": BLUE, "kept for later": ORANGE, "model input": GREY, "model output": GREY}
    fig, ax = plt.subplots(figsize=(13, 9.6), dpi=150)
    style(ax)
    ax.grid(axis="y", visible=False)
    for k, (_, a, b) in enumerate(spans):
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color="#F1EFE9", zorder=0)
    for label, a, b in spans:
        ax.text((a + b) / 2, 1.005, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=11, color=MUTED)
    region_for = {(e["layer_id"], e["tensor_MiB"]): e["region_MiB"] for e in extensions}
    for y, t in enumerate(tensors):
        x0 = -0.45 if t["kind"] == "model input" else t["written_by"] - 0.35
        x1 = t["last_read_by"] + 0.35
        region = region_for.get((t["written_by"], t["MiB"]))
        ax.barh(y, x1 - x0, left=x0, height=0.62, color=color[t["kind"]], zorder=2,
                edgecolor=INK if region else "none", linewidth=2 if region else 0)
        label = f"{t['MiB']:.1f} MiB" if t["MiB"] >= 100 else f"{t['MiB']:.1f}"
        if t["kind"] == "model input":
            label += " (NumPy, not in the arena)"
        note = f"arena takes a new {region:.0f} MiB region" if region else ""   # request found no free piece
        if region and x1 < n - 6:                         # room on the right: size label and note together
            ax.text(x1 + 0.15, y, f"{label}  → {note}", va="center", fontsize=9.5, fontweight="bold", color=INK)
        else:
            ax.text(x1 + 0.15, y, label, va="center", fontsize=9.5, color=INK)
        if region and x1 >= n - 6:
            ax.text(x0 - 0.15, y, note + " →", va="center", ha="right", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_yticks(range(len(tensors)),
                  ["model input" if t["kind"] == "model input" else f"written by {t['written_by']} ({t['writer_op']})"
                   for t in tensors], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.8, n + 0.6)
    ax.set_xticks(range(n))
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("Fused layer ID (ONNX Runtime execution order)", color=INK, fontsize=12)
    ax.set_title("Activation tensors: alive from the layer that writes them to the last layer that reads them",
                 color=INK, fontsize=13, loc="left", pad=26)
    ax.legend(handles=[Patch(color=BLUE, label="Read by the next layer only"),
                       Patch(color=ORANGE, label="Kept for a later layer (skip connection)"),
                       Patch(color=GREY, label="Model input / output")],
              loc="lower left", frameon=False, fontsize=10.5, title="Dark outline: request that made the arena take a new region",
              title_fontsize=10.5, alignment="left")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/exp_a_profile/run_*")
    ap.add_argument("--original", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    args = ap.parse_args()

    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*")))[-1]
    tensors, layers = lifetimes(run_dir, args.original)
    write_csv(os.path.join(run_dir, "tensor_lifetimes.csv"), [{k: v for k, v in t.items() if k != "bytes"} for t in tensors])
    runs = replay(tensors, len(layers))

    measured = None
    logs = sorted(glob.glob(os.path.join(HERE, "results/arena_log_check/run_*/segments.csv")))
    if logs:
        measured = {"file": os.path.relpath(logs[-1], HERE), "segments": read_csv(logs[-1])}
    with open(os.path.join(run_dir, "arena_replay.json"), "w") as f:
        json.dump({"page_bytes": PAGE, "replay": runs, "measured": measured}, f, indent=1)

    for r in runs:
        regions = ", ".join(f"layer {e['layer_id']}: {e['tensor_MiB']} MiB -> region {e['region_MiB']:.0f}"
                            for e in r["new_regions"]) or "none"
        print(f"replay run {r['run']}: new regions [{regions}]; reserved {r['reserved_MiB']} MiB; "
              f"newly written {r['newly_written_MiB']} MiB")
    if measured:
        for s in measured["segments"]:
            if s["segment"] in ("run1", "run2"):
                print(f"log {s['setting']} {s['segment']}: regions [{s['regions_MiB']}], RSS change {s['rss_change_MiB']} MiB")
    plot(tensors, layers, runs[0]["new_regions"], os.path.join(run_dir, "plots", "tensor_lifetimes.png"))


if __name__ == "__main__":
    main()
