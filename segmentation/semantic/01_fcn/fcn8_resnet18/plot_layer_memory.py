#!/usr/bin/env python3
"""Experiment A, extra plot: weight and activation memory per fused layer (counterpart of FlexNN Fig. 1).

Input: layer_stats.csv (layer_stats.py) from one run folder; one row per fused layer that ONNX Runtime
runs, in execution order. Two panels on the same layer-ID axis (separate y scales, not a dual axis):
  - weight memory per layer (weight_MiB)
  - activation memory per layer: the layer's own input + output (act_in_MiB + act_out_MiB, the FlexNN
    per-layer view) as bars, and all tensors alive while the layer runs (live_act_MiB, includes tensors
    kept for skip connections) as a line
Sizes are calculated from tensor shapes (FP32), without kernel temporary buffers (im2col, Winograd).

Output (in the run folder): plots/layer_memory.png

Usage:
  python plot_layer_memory.py                  # latest results/exp_a_profile/run_*
  python plot_layer_memory.py --run-dir results/exp_a_profile/run_20261015_115052
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_profile import BLUE, ORANGE, INK, MUTED, GRID, HERE, read_csv, style

BLOCK_LABEL = {"stem": "Stem", "layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4",
               "fcn_head": "FCN-8s head", "resize": "Resize,\nArgMax", "argmax": "Resize,\nArgMax"}


def block_spans(groups):
    """[(label, first index, last index)] for consecutive layers with the same block label."""
    spans = []
    for i, g in enumerate(groups):
        label = BLOCK_LABEL[g]
        if spans and spans[-1][0] == label:
            spans[-1][2] = i
        else:
            spans.append([label, i, i])
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/exp_a_profile/run_*")
    args = ap.parse_args()

    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*")))[-1]
    rows = sorted(read_csv(os.path.join(run_dir, "layer_stats.csv")), key=lambda r: int(r["order"]))
    x = np.arange(len(rows))
    weights = np.array([float(r["weight_MiB"]) for r in rows])
    own = np.array([float(r["act_in_MiB"]) + float(r["act_out_MiB"]) for r in rows])
    live = np.array([float(r["live_act_MiB"]) for r in rows])
    spans = block_spans([r["group"] for r in rows])

    fig, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(13, 6.4), dpi=150, sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.32})
    for ax in (ax_w, ax_a):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=11)
        for k, (_, a, b) in enumerate(spans):             # light band behind every other block
            if k % 2 == 0:
                ax.axvspan(a - 0.5, b + 0.5, color="#F1EFE9", zorder=0)

    ax_w.bar(x, weights, 0.7, color=BLUE, zorder=2)
    ax_w.set_ylabel("Weights (MiB)", color=INK, fontsize=12)
    ax_w.set_title(f"Weight memory per fused layer (total {weights.sum():.1f} MiB)", color=INK, fontsize=13,
                   loc="left", pad=34)
    top = weights.argmax()
    ax_w.annotate(f"{weights[top]:.1f} MiB: 3×3 conv, 512→512 channels", (x[top], weights[top]),
                  xytext=(-12, -4), textcoords="offset points", ha="right", va="top", fontsize=11, color=INK)
    for label, a, b in spans:                             # block names above the top panel
        ax_w.text((a + b) / 2, 1.04, label, transform=ax_w.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=11, color=MUTED)

    ax_a.bar(x, own, 0.7, color=ORANGE, zorder=2, label="Layer’s own input + output")
    ax_a.plot(x, live, color=INK, linewidth=2, marker="o", markersize=5, zorder=3,
              label="All tensors alive (incl. skip connections)")
    ax_a.set_ylabel("Activations (MiB)", color=INK, fontsize=12)
    ax_a.set_title("Activation memory per fused layer", color=INK, fontsize=13, loc="left")
    ax_a.set_ylim(0, live.max() * 1.18)
    for i, text, dx in ((live.argmax(), f"{live.max():.1f} MiB peak (ArgMax)", -10),
                        (1, f"{live[1]:.1f} MiB (max-pool)", 10)):
        ax_a.annotate(text, (x[i], live[i]), xytext=(dx, 6), textcoords="offset points",
                      ha="right" if dx < 0 else "left", va="bottom", fontsize=11, color=INK)
    ax_a.legend(loc="center", bbox_to_anchor=(0.5, 0.72), frameon=False, fontsize=11)   # empty middle area
    ax_a.set_xticks(x)
    ax_a.set_xlabel("Fused layer ID (ONNX Runtime execution order)", color=INK, fontsize=12)

    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    out = os.path.join(plots, "layer_memory.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
