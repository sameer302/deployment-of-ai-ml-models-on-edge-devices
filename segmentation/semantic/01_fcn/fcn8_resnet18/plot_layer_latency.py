#!/usr/bin/env python3
"""Experiment A, extra plot: measured latency per fused layer, on the same layer-ID axis as layer_memory.png.

Input: analysis_layers.csv (analyze_profile.py) from one run folder; one row per fused layer that ONNX
Runtime runs, in execution order. Bars = median time per layer over all profiled runs, coloured by
roofline bound (compute- or memory-bound); a dark tick on each bar = the roofline minimum (best possible
time at peak compute or full RAM bandwidth), so the part of the bar above the tick is the time lost.
With --plain: one colour, no bound colours or roofline ticks (for slides before the roofline is introduced).

Output (in the run folder): plots/layer_latency.png, or plots/layer_latency_plain.png with --plain

Usage:
  python plot_layer_latency.py                 # latest results/exp_a_profile/run_*
  python plot_layer_latency.py --run-dir results/exp_a_profile/run_20261015_115052
  python plot_layer_latency.py --plain         # time only
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze_profile import BLUE, ORANGE, INK, MUTED, HERE, read_csv, style
from plot_layer_memory import block_spans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/exp_a_profile/run_*")
    ap.add_argument("--plain", action="store_true", help="one colour, no roofline ticks or bound colours")
    args = ap.parse_args()

    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*")))[-1]
    rows = sorted(read_csv(os.path.join(run_dir, "analysis_layers.csv")), key=lambda r: int(r["order"]))
    x = np.arange(len(rows))
    t = np.array([float(r["time_ms"]) for r in rows])
    best = np.array([float(r["roofline_min_ms"]) for r in rows])
    colors = BLUE if args.plain else [BLUE if r["bound"] == "compute" else ORANGE for r in rows]
    spans = block_spans([r["group"] for r in rows])

    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=150)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=11)
    for k, (_, a, b) in enumerate(spans):                 # same bands as layer_memory.png
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color="#F1EFE9", zorder=0)
    for label, a, b in spans:
        ax.text((a + b) / 2, 1.03, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=11, color=MUTED)

    ax.bar(x, t, 0.7, color=colors, zorder=2)
    if not args.plain:
        ax.hlines(best, x - 0.35, x + 0.35, color=INK, linewidth=2.5, zorder=3)
    for i, dx, ha in ((0, 10, "left"), (len(rows) - 1, -10, "right")):
        ax.annotate(f"{t[i]:.1f} ms", (x[i], t[i]), xytext=(dx, 2), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=11, color=INK)
    ax.set_ylim(0, t.max() * 1.12)
    ax.set_ylabel("Median time (ms)", color=INK, fontsize=12)
    ax.set_title(f"Latency per fused layer (sum {t.sum():,.0f} ms)", color=INK, fontsize=13, loc="left", pad=34)
    ax.set_xticks(x)
    ax.set_xlabel("Fused layer ID (ONNX Runtime execution order)", color=INK, fontsize=12)
    if not args.plain:
        ax.legend(handles=[Patch(color=BLUE, label="Compute-bound layer"),
                           Patch(color=ORANGE, label="Memory-bound layer"),
                           Line2D([], [], color=INK, linewidth=2.5, label="Best possible (roofline minimum)")],
                  loc="upper center", bbox_to_anchor=(0.62, 0.98), frameon=False, fontsize=11, ncol=3)

    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    out = os.path.join(plots, "layer_latency_plain.png" if args.plain else "layer_latency.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
