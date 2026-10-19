#!/usr/bin/env python3
"""Experiment A, script 4: join measured layer times with layer computation / memory, roofline, time gap.

Inputs from one run folder: per_layer.csv and per_run.csv (profile_layers.py), layer_stats.csv
(layer_stats.py). Hardware limits: compute peak = 16 FP32 FLOPs/cycle/core (2 NEON FMA units x 4 lanes
x 2) x 2.4 GHz x 4 cores = 153.6 GFLOPS; RAM bandwidth from STREAM with 4 threads (8.2 GB/s,
common/bench/stream/README.md).

Per layer (roofline model):
  compute floor  = FLOPs / peak             memory floor = bytes moved / bandwidth
  roofline min   = max(compute floor, memory floor)
  bound          = "memory" if intensity < ridge (peak / bandwidth) else "compute"
  lost           = measured time - roofline min

Time gap (sums over layers, medians): measured = ideal compute + memory floor extra
  + compute-bound kernels below peak + memory-bound kernels below the RAM limit; framework overhead
  (wall - sum of layers) is reported separately.

Outputs (in the run folder): analysis_layers.csv, analysis_blocks.csv, analysis.json,
  plots/roofline.png, plots/time_vs_compute.png, plots/time_gap.png

Usage:
  python analyze_profile.py                         # latest results/exp_a_profile/run_*
  python analyze_profile.py --bandwidth-gbs 8.2 --peak-gflops 153.6
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIB = 2 ** 20
BLUE, ORANGE = "#2A78D6", "#EB6834"            # same pair as plot_results.py
INK, MUTED, GRID = "#1B2333", "#5E6574", "#E2DFD6"
GROUP_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "fcn_head", "resize", "argmax"]
GROUP_LABEL = {"stem": "Stem", "layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4",
               "fcn_head": "FCN head", "resize": "Resize", "argmax": "ArgMax"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/exp_a_profile/run_*")
    ap.add_argument("--peak-gflops", type=float, default=153.6)
    ap.add_argument("--bandwidth-gbs", type=float, default=8.2, help="STREAM, 4 threads")
    args = ap.parse_args()

    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*")))[-1]
    times = {r["layer"]: float(r["median_ms"]) for r in read_csv(os.path.join(run_dir, "per_layer.csv"))}
    stats = read_csv(os.path.join(run_dir, "layer_stats.csv"))
    per_run = read_csv(os.path.join(run_dir, "per_run.csv"))
    peak, bw = args.peak_gflops * 1e9, args.bandwidth_gbs * 1e9
    ridge = peak / bw

    rows = []
    for s in stats:
        t = times[s["layer"]] / 1e3
        flops, moved = int(s["FLOPs"]), float(s["bytes_moved_MiB"]) * MIB
        compute_floor, memory_floor = flops / peak, moved / bw
        intensity = flops / moved
        rows.append({
            "order": int(s["order"]), "layer": s["layer"], "op_type": s["op_type"], "group": s["group"],
            "fused_from": s["fused_from"], "time_ms": round(t * 1e3, 3),
            "GFLOPs": round(flops / 1e9, 4), "MiB_moved": round(moved / MIB, 2),
            "intensity_FLOP_per_byte": round(intensity, 2),
            "bound": "memory" if intensity < ridge else "compute",
            "achieved_GFLOPS": round(flops / t / 1e9, 2), "pct_of_peak": round(100 * flops / t / peak, 1),
            "achieved_GBps": round(moved / t / 1e9, 2), "pct_of_bandwidth": round(100 * moved / t / bw, 1),
            "compute_floor_ms": round(compute_floor * 1e3, 3), "memory_floor_ms": round(memory_floor * 1e3, 3),
            "roofline_min_ms": round(max(compute_floor, memory_floor) * 1e3, 3),
            "lost_ms": round((t - max(compute_floor, memory_floor)) * 1e3, 3),
        })

    total = sum(r["time_ms"] for r in rows)
    total_flops = sum(r["GFLOPs"] for r in rows)
    blocks = []
    for g in GROUP_ORDER:
        rs = [r for r in rows if r["group"] == g]
        t, f = sum(r["time_ms"] for r in rs), sum(r["GFLOPs"] for r in rs)
        blocks.append({"group": g, "time_ms": round(t, 2), "time_share_pct": round(100 * t / total, 2),
                       "GFLOPs": round(f, 3), "FLOPs_share_pct": round(100 * f / total_flops, 2),
                       "time_over_FLOPs_share": round((t / total) / (f / total_flops), 2),
                       "achieved_GFLOPS": round(f / t * 1e3, 1), "lost_ms": round(sum(r["lost_ms"] for r in rs), 2)})

    ideal = sum(r["compute_floor_ms"] for r in rows)
    mem_extra = sum(max(0.0, r["memory_floor_ms"] - r["compute_floor_ms"]) for r in rows)
    lost_c = sum(r["lost_ms"] for r in rows if r["bound"] == "compute")
    lost_m = sum(r["lost_ms"] for r in rows if r["bound"] == "memory")
    framework = float(np.median([float(r["wall_ms"]) - float(r["sum_layers_ms"]) for r in per_run]))
    gap = [("Ideal: all FLOPs at peak", ideal),
           ("RAM floor of memory-bound layers", mem_extra),
           ("Compute-bound kernels below peak", lost_c),
           ("Memory-bound kernels below RAM limit", lost_m)]
    summary = {
        "run_dir": os.path.relpath(run_dir, HERE),
        "limits": {"peak_GFLOPS": args.peak_gflops, "bandwidth_GBps": args.bandwidth_gbs,
                   "ridge_FLOP_per_byte": round(ridge, 1)},
        "sum_of_layer_medians_ms": round(total, 1),
        "time_gap_ms": {k: round(v, 1) for k, v in gap},
        "time_gap_pct_of_measured": {k: round(100 * v / total, 1) for k, v in gap},
        "framework_overhead_ms_median": round(framework, 2),
        "overall_GFLOPS": round(total_flops / total * 1e3, 1),
        "overall_pct_of_peak": round(100 * total_flops / total * 1e3 / args.peak_gflops, 1),
        "memory_bound_layers": [r["layer"] for r in rows if r["bound"] == "memory"],
        "top_lost_ms": [{k: r[k] for k in ("layer", "group", "bound", "time_ms", "roofline_min_ms", "lost_ms")}
                        for r in sorted(rows, key=lambda r: -r["lost_ms"])[:8]],
    }
    write_csv(os.path.join(run_dir, "analysis_layers.csv"), rows)
    write_csv(os.path.join(run_dir, "analysis_blocks.csv"), blocks)
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    # 1. Roofline: achieved GFLOPS vs arithmetic intensity, log-log.
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    x = np.logspace(-1.3, 3, 200)
    ax.plot(x, np.minimum(args.peak_gflops, x * args.bandwidth_gbs), color=INK, linewidth=2)
    ax.text(250, args.peak_gflops * 1.08, f"compute peak {args.peak_gflops:.0f} GFLOPS", color=INK, fontsize=10, ha="center")
    ax.text(0.06, 9, f"RAM limit {args.bandwidth_gbs} GB/s\n(STREAM, 4 threads)", color=INK, fontsize=10)
    ax.axvline(ridge, color=MUTED, linewidth=1, linestyle=":")
    ax.text(ridge * 0.92, 200, f"ridge {ridge:.0f} FLOP/B", color=MUTED, fontsize=9, ha="right")
    for bound, color, marker, label in (("compute", BLUE, "o", "Compute-bound layer"),
                                        ("memory", ORANGE, "^", "Memory-bound layer")):
        rs = [r for r in rows if r["bound"] == bound]
        ax.scatter([r["intensity_FLOP_per_byte"] for r in rs], [r["achieved_GFLOPS"] for r in rs],
                   s=[24 + 2.2 * r["time_ms"] for r in rs], color=color, marker=marker, alpha=0.9,
                   edgecolors="white", linewidths=1.5, label=label, zorder=3)
    notes = {"Conv_0": "stem 7×7 conv", "MaxPool_3": "MaxPool", "Resize_81": "Resize", "ArgMax_82": "ArgMax",
             "Conv_43": "layer3 3×3 conv", "Conv_23": "1×1 downsample", "Conv_73": "score_8 + add"}
    offsets = {"Conv_43": (-40, -24)}
    for r in rows:
        if r["layer"] in notes:
            ax.annotate(notes[r["layer"]], (r["intensity_FLOP_per_byte"], r["achieved_GFLOPS"]),
                        textcoords="offset points", xytext=offsets.get(r["layer"], (8, -12)), fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 1000)
    ax.set_ylim(0.1, 300)
    ax.set_xlabel("Arithmetic intensity (FLOPs per byte moved)", color=INK)
    ax.set_ylabel("Achieved GFLOPS", color=INK)
    ax.set_title("FCN8-ResNet18 layers on the Pi 5 roofline (ONNX Runtime FP32, 4 threads); marker area ∝ time",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "roofline.png"))
    plt.close(fig)

    # 2. Share of time vs share of FLOPs per block.
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    y = np.arange(len(blocks))[::-1]
    h = 0.38
    ax.barh(y + h / 2, [b["time_share_pct"] for b in blocks], h - 0.04, color=BLUE, label="Share of time")
    ax.barh(y - h / 2, [b["FLOPs_share_pct"] for b in blocks], h - 0.04, color=ORANGE, label="Share of FLOPs")
    for yi, b in zip(y, blocks):
        ax.text(b["time_share_pct"] + 0.4, yi + h / 2, f"{b['time_share_pct']:.1f}%", va="center", fontsize=9, color=INK)
        ax.text(b["FLOPs_share_pct"] + 0.4, yi - h / 2, f"{b['FLOPs_share_pct']:.2f}%", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y, [GROUP_LABEL[b["group"]] for b in blocks], color=INK)
    ax.set_xlabel("% of total", color=INK)
    ax.set_title("Where the time goes vs where the computation is", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "time_vs_compute.png"))
    plt.close(fig)

    # 3. Time gap: from ideal compute time to measured time.
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    labels = [k for k, _ in gap] + ["Measured (sum of layer medians)"]
    vals = [v for _, v in gap] + [total]
    lefts = [0, gap[0][1], gap[0][1] + gap[1][1], gap[0][1] + gap[1][1] + gap[2][1], 0]
    colors = [BLUE, ORANGE, BLUE, ORANGE, INK]
    yy = np.arange(len(vals))[::-1]
    ax.barh(yy, vals, 0.62, left=lefts, color=colors)
    for yi, l, v in zip(yy, lefts, vals):
        ax.text(l + v + 15, yi, f"{v:.0f} ms ({100 * v / total:.0f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(yy, labels, color=INK, fontsize=10)
    ax.set_xlim(0, total * 1.22)
    ax.set_xlabel("ms per image", color=INK)
    ax.set_title("From the ideal 0.93 s to the measured time", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "time_gap.png"))
    plt.close(fig)

    print(f"ridge {ridge:.1f} FLOP/B | overall {summary['overall_GFLOPS']} GFLOPS ({summary['overall_pct_of_peak']}% of peak) | "
          f"measured {total:.1f} ms | framework overhead {framework:.2f} ms")
    for k, v in gap:
        print(f"  {k:<40} {v:8.1f} ms  {100 * v / total:5.1f}%")
    print(f"memory-bound layers: {', '.join(summary['memory_bound_layers'])}")
    print("block       time%   FLOPs%  ratio  GFLOPS  lost ms")
    for b in blocks:
        print(f"  {b['group']:<9} {b['time_share_pct']:6.2f} {b['FLOPs_share_pct']:7.2f} {b['time_over_FLOPs_share']:6.2f} "
              f"{b['achieved_GFLOPS']:7.1f} {b['lost_ms']:8.1f}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
