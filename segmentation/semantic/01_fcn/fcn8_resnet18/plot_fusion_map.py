#!/usr/bin/env python3
"""Experiment A, extra diagram: how the 57 ONNX layers become the 30 fused layers ONNX Runtime runs, per block.

Input: layer_stats.csv (layer_stats.py) from one run folder; its `fused_from` column lists the original
ONNX layers inside each fused layer (e.g. Conv_7+Add_9+Relu_10). One column per block (stem, layer1-4,
FCN-8s head, Resize + ArgMax); each original layer is a box, and the boxes of one fused layer are framed together
with the fused layer ID (execution order, same IDs as layer_latency.png / layer_memory.png). Original
layers are listed in fused execution order, so a 1x1 downsample conv sits between the two convs of its
residual block.

Output (in the run folder): plots/fusion_map.png

Usage:
  python plot_fusion_map.py                    # latest results/exp_a_profile/run_*
  python plot_fusion_map.py --run-dir results/exp_a_profile/run_20261015_115052
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from analyze_profile import BLUE, INK, MUTED, HERE, read_csv

COLUMNS = [("stem", "Stem"), ("layer1", "layer1"), ("layer2", "layer2"), ("layer3", "layer3"),
           ("layer4", "layer4"), ("fcn_head", "FCN-8s head"), ("output", "Resize +\nArgMax")]
SHORT = {"BatchNormalization": "BN", "ConvTranspose": "ConvT"}
FUSED_FILL, PLAIN_FILL, PLAIN_EDGE, BOX_EDGE = "#E3EEFB", "#F1EFE9", "#8A93A3", "#C9CEDA"
ENC_COLOR, DEC_COLOR = "#1C5CAB", "#B4461A"                # encoder / decoder labels, as in the deck


def short(name):
    op, num = name.rsplit("_", 1)
    return f"{SHORT.get(op, op)}_{num}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/exp_a_profile/run_*")
    args = ap.parse_args()

    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*")))[-1]
    rows = sorted(read_csv(os.path.join(run_dir, "layer_stats.csv")), key=lambda r: int(r["order"]))
    groups = {c[0]: [] for c in COLUMNS}                  # column -> [(fused id, [original names])]
    for r in rows:
        col = "output" if r["group"] in ("resize", "argmax") else r["group"]
        groups[col].append((int(r["order"]), r["fused_from"].split("+")))

    fig, ax = plt.subplots(figsize=(12, 6.9), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.15, 7.7)
    ax.axis("off")
    col_w, step, box_h, gap = 12 / len(COLUMNS), 0.43, 0.34, 0.09
    n_enc = [c[0] for c in COLUMNS].index("fcn_head")    # encoder = columns before the FCN-8s head
    for x_a, x_b, label, color in ((0.1, n_enc * col_w - 0.1, "ENCODER: ResNet-18", ENC_COLOR),
                                   (n_enc * col_w + 0.1, 12 - 0.02, "DECODER", DEC_COLOR)):
        ax.plot([x_a, x_a, x_b, x_b], [7.02, 7.12, 7.12, 7.02], color=color, linewidth=2)
        ax.text((x_a + x_b) / 2, 7.3, label, ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax.plot([n_enc * col_w - 0.03] * 2, [0.75, 6.0], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    for i, (key, title) in enumerate(COLUMNS):
        x0 = i * col_w
        n_parts = sum(len(p) for _, p in groups[key])
        ax.text(x0 + col_w / 2 + 0.12, 6.4, title, ha="center", va="baseline", linespacing=1.05, fontsize=13, fontweight="bold",
                color=INK)
        ax.text(x0 + col_w / 2 + 0.12, 6.2, f"{n_parts} → {len(groups[key])} layers", ha="center", va="center",
                fontsize=11.5, color=MUTED)
        y = 5.75
        for fid, parts in groups[key]:
            top = y + box_h / 2 + 0.05
            bottom = y - (len(parts) - 1) * step - box_h / 2 - 0.05
            merged = len(parts) > 1
            ax.add_patch(FancyBboxPatch((x0 + 0.42, bottom), col_w - 0.5, top - bottom,
                                        boxstyle="round,pad=0,rounding_size=0.06", linewidth=1.6,
                                        facecolor=FUSED_FILL if merged else PLAIN_FILL,
                                        edgecolor=BLUE if merged else PLAIN_EDGE))
            ax.add_patch(Circle((x0 + 0.22, (top + bottom) / 2), 0.16, color=INK))
            ax.text(x0 + 0.22, (top + bottom) / 2, str(fid), ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color="white")
            for name in parts:
                ax.add_patch(FancyBboxPatch((x0 + 0.52, y - box_h / 2), col_w - 0.7, box_h,
                                            boxstyle="round,pad=0,rounding_size=0.04", linewidth=0.8,
                                            facecolor="white", edgecolor=BOX_EDGE))
                ax.text(x0 + 0.52 + (col_w - 0.7) / 2, y, short(name), ha="center", va="center", fontsize=10.5,
                        color=INK)
                y -= step
            y -= gap

    legend_y = 0.45
    ax.add_patch(FancyBboxPatch((0.1, legend_y - 0.14), 0.4, 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=FUSED_FILL, edgecolor=BLUE, linewidth=1.6))
    ax.text(0.6, legend_y, "Merged by ONNX Runtime into one kernel", va="center", fontsize=11.5, color=INK)
    ax.add_patch(FancyBboxPatch((5.4, legend_y - 0.14), 0.4, 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=PLAIN_FILL, edgecolor=PLAIN_EDGE, linewidth=1.6))
    ax.text(5.9, legend_y, "Runs unchanged", va="center", fontsize=11.5, color=INK)
    ax.add_patch(Circle((8.05, legend_y), 0.16, color=INK))
    ax.text(8.05, legend_y, "n", ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
    ax.text(8.3, legend_y, f"Fused layer ID (execution order)",
            va="center", fontsize=11.5, color=INK)

    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    out = os.path.join(plots, "fusion_map.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
