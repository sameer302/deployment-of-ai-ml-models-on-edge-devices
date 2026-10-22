"""Architecture diagrams + layer table for FCN8-ResNet18, built from the ONNX file itself.

Every shape, parameter count and MAC count comes from the model (ONNX shape inference),
not from the paper. The block layout (which node indices form which block) is specific
to this export; the asserts below fail loudly if the graph ever differs.

Outputs (next to this script):
  fcn8_resnet18_architecture.png / .svg   block diagram: encoder, skips, decoder, with tensor shapes
  resnet_basicblock.png / .svg            the two residual block types used in the encoder
  layer_table.csv                         every layer: block, op, kernel, stride, in/out shape, params, MACs

Usage (from repo root):
  venv_cpu/bin/python segmentation/semantic/01_fcn/fcn8_resnet18/architecture/draw_architecture.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from onnx import numpy_helper, shape_inference  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "../models/fcn8_resnet_v1_18.onnx")

# Reference palette (dataviz skill): one hue per role, fixed order; text always in ink colors.
ENC, SKIP, DEC, POST = "#2a78d6", "#1baf7a", "#eb6834", "#eda100"
INK, INK2, MUTED, IO_FILL, SURFACE = "#0b0b0b", "#52514e", "#898781", "#f0efec", "#fcfcfb"

# Node index ranges of each block in this export (see layer_table.csv).
BLOCKS = [
    ("stem", range(0, 3)), ("layer1", range(3, 13)), ("layer2", range(13, 24)),
    ("layer3", range(24, 35)), ("layer4", range(35, 46)),
    ("score_32", range(46, 47)), ("up_32to16", range(47, 49)), ("score_16", range(49, 50)), ("add_16", range(50, 51)),
    ("up_16to8", range(51, 53)), ("score_8", range(53, 54)), ("add_8", range(54, 55)),
    ("resize_x8", range(55, 56)), ("argmax", range(56, 57)),
]


def analyse(path):
    model = onnx.load(path)
    graph = shape_inference.infer_shapes(model).graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(graph.value_info) + list(graph.input) + list(graph.output)}
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    block_of = {i: name for name, r in BLOCKS for i in r}
    assert len(model.graph.node) == 57, "graph differs from the export this layout was written for"

    rows = []
    for i, n in enumerate(model.graph.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        data_in = [x for x in n.input if x and x not in weights]
        out_shape = shapes.get(n.output[0])
        w = weights.get(n.input[1]) if len(n.input) > 1 else None
        if n.op_type == "Conv":
            macs = int(np.prod(out_shape) * np.prod(w.shape[1:]))
        elif n.op_type == "ConvTranspose":
            macs = int(np.prod(shapes[data_in[0]]) * np.prod(w.shape[1:]))
        else:
            macs = 0
        # Resize's only "weights" are shape constants, not learned parameters.
        params = 0 if n.op_type == "Resize" else int(sum(weights[x].size for x in n.input if x in weights))
        rows.append({
            "index": i, "block": block_of[i], "op": n.op_type,
            "kernel": "x".join(map(str, attrs["kernel_shape"])) if "kernel_shape" in attrs else "",
            "stride": "x".join(map(str, attrs["strides"])) if "strides" in attrs else "",
            "input_shape": " + ".join("x".join(map(str, shapes[x][1:])) for x in data_in),
            "output_shape": "x".join(map(str, out_shape[1:])),
            "params": params, "MACs": macs,
        })
    return rows


def block_stats(rows):
    stats = {}
    for name, r in BLOCKS:
        sel = [rows[i] for i in r]
        stats[name] = {"params": sum(x["params"] for x in sel), "MACs": sum(x["MACs"] for x in sel),
                       "out": sel[-1]["output_shape"].replace("x", "×")}
    return stats


def fmt_params(p):
    return f"{p/1e6:.2f} M params" if p >= 1e6 else f"{p/1e3:.1f} K params"


def fmt_macs(m, total):
    return f"{m/1e9:.1f} GMACs ({100*m/total:.0f}%)" if m else "no MACs"


def box(ax, x, y, w, h, color, title, lines, fill=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill or matplotlib.colors.to_rgba(color, 0.14), ec=color, lw=1.6))
    ax.text(x, y + h / 2 - 0.16, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
    ax.text(x, y + h / 2 - 0.45, "\n".join(lines), ha="center", va="top", fontsize=8, color=INK2, linespacing=1.45)


def arrow(ax, p0, p1, color=INK2, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def draw_main(stats, total_params, total_macs, out_base):
    fig, ax = plt.subplots(figsize=(18, 9.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-1.3, 15.4)
    ax.set_ylim(-1.5, 8.4)
    ax.axis("off")

    xs = [0, 2.6, 5.2, 7.8, 10.4, 13.0]
    top, mid, bot = 5.75, 3.35, 1.0
    w, h, hs = 2.3, 1.75, 1.05

    # Resolution / stage headers
    for x, lab in zip(xs, ["full res (1/1)", "1/4", "1/4", "1/8", "1/16", "1/32"]):
        ax.text(x, 7.05, lab, ha="center", fontsize=9, color=MUTED)
    ax.text(-1.2, 7.75, "ENCODER  (ResNet-18, BatchNorm folded into conv weights)", fontsize=10, color=ENC, fontweight="bold")
    ax.text(-1.2, -1.2, "DECODER  (FCN-8s head, then Resize + ArgMax; 19 Cityscapes classes)", fontsize=10, color=DEC, fontweight="bold")

    s = stats
    t = total_macs
    box(ax, xs[0], top, w, h, MUTED, "Input image", ["RGB, mean/std normalized", "3×1024×1920", "(Cityscapes 2048×1024", "resized to 1920 wide)"], fill=IO_FILL)
    box(ax, xs[1], top, w, h, ENC, "Stem", ["7×7 conv, stride 2 + ReLU", "3×3 max-pool, stride 2",
                                         s["stem"]["out"], fmt_params(s["stem"]["params"]), fmt_macs(s["stem"]["MACs"], t)])
    box(ax, xs[2], top, w, h, ENC, "layer1", ["2 × BasicBlock, 64 ch", "identity skips",
                                           s["layer1"]["out"], fmt_params(s["layer1"]["params"]), fmt_macs(s["layer1"]["MACs"], t)])
    for i, (name, ch) in enumerate([("layer2", 128), ("layer3", 256), ("layer4", 512)]):
        box(ax, xs[3 + i], top, w, h, ENC, name, [f"2 × BasicBlock, {ch} ch", "1st block stride 2 (1×1 proj.)",
                                                 s[name]["out"], fmt_params(s[name]["params"]), fmt_macs(s[name]["MACs"], t)])
    for i in range(5):
        arrow(ax, (xs[i] + w / 2, top), (xs[i + 1] - w / 2, top))

    # Score (1×1 conv) branches: one per tapped encoder stage
    taps = [(xs[3], "score_8", "128→19"), (xs[4], "score_16", "256→19"), (xs[5], "score_32", "512→19")]
    for x, name, ch in taps:
        box(ax, x, mid, w, hs, SKIP, "1×1 conv score", [f"{ch} ch,  {s[name]['out']}", fmt_params(s[name]["params"])])
        arrow(ax, (x, top - h / 2), (x, mid + hs / 2), color=SKIP)

    # Decoder: upsample ×2 (ConvT + BN), then add the skip scores
    box(ax, xs[4], bot, w, h * 0.92, DEC, "Upsample ×2  +  add", ["4×4 ConvTranspose s2 + BN", "+ score_16 (skip from layer3)",
                                                               s["add_16"]["out"], fmt_params(s["up_32to16"]["params"])])
    box(ax, xs[3], bot, w, h * 0.92, DEC, "Upsample ×2  +  add", ["4×4 ConvTranspose s2 + BN", "+ score_8 (skip from layer2)",
                                                               s["add_8"]["out"], fmt_params(s["up_16to8"]["params"])])
    box(ax, xs[2], bot, w, h * 0.92, POST, "Resize ×8", ["bilinear (align_corners)", "no learned weights",
                                                     s["resize_x8"]["out"], "19 class scores / pixel"])
    box(ax, xs[1], bot, w, h * 0.92, POST, "ArgMax", ["over the 19 classes", "(inside the ONNX model)", s["argmax"]["out"]])
    box(ax, xs[0], bot, w, h * 0.92, MUTED, "Output", ["class map, int64", "1×1024×1920", "→ resized to 2048×1024", "for evaluation"], fill=IO_FILL)

    # FCN-8s head = the 3 score convs + the 2 upsample-and-add steps (fusion map: FCN-8s head, IDs 21-27)
    hx0, hx1, hy0, hy1 = xs[3] - w / 2 - 0.12, xs[5] + w / 2 + 0.12, bot - h * 0.46 - 0.12, mid + hs / 2 + 0.12
    ax.add_patch(FancyBboxPatch((hx0, hy0), hx1 - hx0, hy1 - hy0, boxstyle="round,pad=0,rounding_size=0.12",
                                fill=False, ec=INK2, lw=1.4, ls=(0, (5, 3))))
    ax.text(xs[5] + 0.35, bot - 0.15, "FCN-8s head", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(xs[5] + 0.35, bot - 0.5, "3 score convs +\n2 upsample-and-add", ha="center", va="center", fontsize=9, color=INK2)

    arrow(ax, (xs[5], mid - hs / 2), (xs[4] + w / 2, bot + 0.25), color=DEC, rad=-0.25)
    arrow(ax, (xs[4], mid - hs / 2), (xs[4], bot + h * 0.46), color=SKIP)
    arrow(ax, (xs[3], mid - hs / 2), (xs[3], bot + h * 0.46), color=SKIP)
    for a, b in [(4, 3), (3, 2), (2, 1), (1, 0)]:
        arrow(ax, (xs[a] - w / 2, bot), (xs[b] + w / 2, bot), color=DEC if a > 2 else POST)

    ax.legend(handles=[Patch(fc=matplotlib.colors.to_rgba(c, 0.14), ec=c, lw=1.6, label=l) for c, l in
                       [(ENC, "Encoder (feature extraction, downsampling)"), (SKIP, "Skip / score branches (1×1 conv to 19 classes)"),
                        (DEC, "Decoder upsampling + fusion (learned)"), (POST, "Final upsampling + class decision (no weights)")]],
              loc="lower right", bbox_to_anchor=(1.0, -0.02), frameon=False, fontsize=9)
    ax.set_title(f"FCN8-ResNet18 (Hailo fcn8_resnet_v1_18): {total_params/1e6:.2f} M params, "
                 f"{total_macs/1e9:.1f} GMACs ≈ {2*total_macs/1e9:.0f} GFLOPs per 1024×1920 image",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    plt.close(fig)


def draw_basicblock(out_base):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2))
    fig.patch.set_facecolor(SURFACE)
    titles = ["BasicBlock, identity skip\n(layer1, and 2nd block of every stage)",
              "BasicBlock, downsampling skip\n(1st block of layer2, layer3, layer4)"]
    for ax, title, down in zip(axes, titles, [False, True]):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-2.0, 2.9)
        ax.set_ylim(-0.6, 7.2)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="center")
        ys = [6.4, 5.0, 3.6, 2.2, 1.0, 0.0]
        labels = ["x  (C×H×W)", "3×3 conv" + (", stride 2" if down else "") + "  + ReLU",
                  "3×3 conv", "⊕  add", "ReLU", "out  (" + ("2C×H/2×W/2" if down else "C×H×W") + ")"]
        for y, lab in zip(ys, labels):
            is_io = lab.startswith(("x", "out"))
            ax.add_patch(FancyBboxPatch((-1.15, y - 0.3), 2.3, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=IO_FILL if is_io else matplotlib.colors.to_rgba(ENC, 0.14),
                                        ec=MUTED if is_io else ENC, lw=1.4))
            ax.text(0, y, lab, ha="center", va="center", fontsize=9.5, color=INK)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            arrow(ax, (0, y0 - 0.3), (0, y1 + 0.3))
        # Skip path: identity or 1×1 projection conv, joining at the add
        if down:
            ax.add_patch(FancyBboxPatch((1.4, 3.75), 1.2, 1.3, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=matplotlib.colors.to_rgba(SKIP, 0.14), ec=SKIP, lw=1.4))
            ax.text(2.0, 4.4, "1×1 conv\nstride 2\n(projection:\nmatches shape)", ha="center", va="center",
                    fontsize=8, color=INK, linespacing=1.3)
            arrow(ax, (1.15, 6.4), (2.0, 5.05), color=SKIP, rad=-0.2)
            arrow(ax, (2.0, 3.75), (1.15, 2.2), color=SKIP, rad=-0.2)
        else:
            arrow(ax, (1.15, 6.4), (1.15, 2.2), color=SKIP, rad=-0.45)
            ax.text(2.55, 4.3, "identity\n(x passes\nunchanged)", ha="center", va="center", fontsize=8.5, color=INK2)
    fig.text(0.5, 0.015, "BatchNorm layers were folded into the conv weights at export, so each conv above = conv + BN.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=200)
    plt.close(fig)


def main():
    rows = analyse(MODEL)
    with open(os.path.join(HERE, "layer_table.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    stats = block_stats(rows)
    total_params = sum(r["params"] for r in rows)
    total_macs = sum(r["MACs"] for r in rows)
    draw_main(stats, total_params, total_macs, os.path.join(HERE, "fcn8_resnet18_architecture"))
    draw_basicblock(os.path.join(HERE, "resnet_basicblock"))

    print(f"total: {total_params/1e6:.3f} M params, {total_macs/1e9:.2f} GMACs")
    for name, _ in BLOCKS:
        st = stats[name]
        print(f"  {name:<10} out {st['out']:<16} {st['params']:>10,} params  {st['MACs']/1e9:6.2f} GMACs ({100*st['MACs']/total_macs:4.1f}%)")


if __name__ == "__main__":
    main()
