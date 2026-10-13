"""Plot the outputs of run_eval.py into slide-ready PNGs + a summary table CSV.

Usage (from repo root):
  venv_cpu/bin/python segmentation/semantic/01_fcn/fcn8_resnet18/plot_results.py <results_dir>
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots in fixed order + neutral inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
HAILO_REF_MIOU = 69.4  # Hailo Model Zoo, fcn8_resnet_v1_18 float mIoU on Cityscapes
MIB = 2 ** 20

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.size": 10, "text.color": INK2,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def per_class_iou(rows, s, out):
    rows = sorted([r for r in rows if r["iou"]], key=lambda r: float(r["iou"]))
    names, vals = [r["class"] for r in rows], [100 * float(r["iou"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(names, vals, color=BLUE, height=0.65)
    for y, v in enumerate(vals):
        ax.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=9, color=INK2)
    miou = s["accuracy"]["mIoU"]
    ax.axvline(miou, color=INK, lw=1.2, ls="--")
    ax.axvline(HAILO_REF_MIOU, color=MUTED, lw=1.2, ls=":")
    # Labels sit either side of the two (close) lines, below the bars.
    left, right = sorted([(miou, f"ours {miou:.1f} ", INK), (HAILO_REF_MIOU, f" Hailo ref {HAILO_REF_MIOU}", MUTED)])
    ax.text(left[0], -1.0, f"{left[1].strip()} ", color=left[2], fontsize=9, va="center", ha="right")
    ax.text(right[0], -1.0, f" {right[1].strip()}", color=right[2], fontsize=9, va="center", ha="left")
    ax.set_xlim(0, 105)
    ax.set_ylim(-1.6, len(vals))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("IoU (%)")
    ax.set_title(f"Per-class IoU, Cityscapes val ({s['images_evaluated']} images), ORT FP32 on Pi 5 CPU")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def latency_breakdown(s, out):
    stages = [("load_ms", "Image + label load"), ("pre_ms", "Pre-process"), ("infer_ms", "Inference"),
              ("post_ms", "Post-process"), ("total_ms", "End-to-end")]
    labels = [l for _, l in stages]
    means = [s["latency_ms"][k]["mean"] for k, _ in stages]
    stds = [s["latency_ms"][k]["std"] for k, _ in stages]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, means, xerr=stds, color=BLUE, height=0.6, error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    for y, (m, sd) in enumerate(zip(means, stds)):
        ax.text(m + sd + max(means) * 0.01, y, f"{m:,.0f} ± {sd:,.0f} ms", va="center", fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(means) * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Latency per image (ms, mean ± std)")
    t = s["throughput"]
    ax.set_title(f"Latency per stage: {t['fps_end_to_end']:.2f} FPS end-to-end, "
                 f"{t['fps_inference_only']:.2f} FPS inference-only")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def latency_per_image(rows, out):
    idx = list(range(1, len(rows) + 1))
    total = [float(r["total_ms"]) for r in rows]
    infer = [float(r["infer_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, total, color=BLUE, lw=1.5, label="End-to-end")
    ax.plot(idx, infer, color=ORANGE, lw=1.5, label="Inference only")
    ax.text(idx[-1], total[-1], "  end-to-end", color=INK2, fontsize=9, va="bottom")
    ax.text(idx[-1], infer[-1], "  inference", color=INK2, fontsize=9, va="top")
    ax.set_ylim(0, max(total) * 1.15)
    ax.set_xlim(1, len(idx) * 1.08)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Latency per image across the run (stability / thermal check)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def to_mib(container, name):
    """Memory field `name` in MiB. Runs before 2026-10-13 logged MB (10^6 bytes) as `<name>_MB`;
    newer runs log MiB (2^20 bytes, as `free -m`) as `<name>_MiB`."""
    if f"{name}_MiB" in container:
        return container[f"{name}_MiB"]
    v = container[f"{name}_MB"]
    if isinstance(v, dict):  # summary.json stats block
        return {k: round(x * 1e6 / MIB, 2) for k, x in v.items()}
    return round(float(v) * 1e6 / MIB, 1)


def platform_timeline(rows, out):
    t = [float(r["elapsed_s"]) / 60 for r in rows]
    for r in rows:
        r["process_rss_MiB"] = to_mib(r, "process_rss")
    panels = [("cpu_total_percent", "CPU utilisation (%)", (0, 105)),
              ("soc_temp_C", "SoC temperature (°C)", None),
              ("cpu_freq_MHz", "CPU frequency (MHz)", (0, 2800)),
              ("process_rss_MiB", "Process memory RSS (MiB)", "zero")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 9), sharex=True)
    for ax, (key, label, ylim) in zip(axes, panels):
        vals = [float(r[key]) if r[key] not in ("", "None") else float("nan") for r in rows]
        ax.plot(t, vals, color=BLUE, lw=1.5)
        ax.set_ylabel(label, fontsize=9)
        if ylim == "zero":
            ax.set_ylim(0, max(vals) * 1.2)
        elif ylim:
            ax.set_ylim(*ylim)
        else:
            lo, hi = min(vals), max(vals)
            pad = max((hi - lo) * 0.2, 1)
            ax.set_ylim(lo - pad, hi + pad)
    axes[-1].set_xlabel("Time since start of evaluation (min)")
    axes[0].set_title("Platform metrics during evaluation (Pi 5 CPU, 1 s sampling)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def summary_table(s, out):
    lat, p, a = s["latency_ms"], s["platform"], s["accuracy"]
    rows = [
        ("Model", "FCN8-ResNet18"), ("Runtime / precision", f"{s['runtime']} / {s['precision']}"),
        ("Input size", f"{s['input_size'][1]}x{s['input_size'][0]}"), ("Parameters (M)", s["params_M"]),
        ("Dataset", f"{s['dataset']} ({s['images_evaluated']} images)"),
        ("mIoU (%)", a["mIoU"]), ("Hailo reference mIoU, float (%)", HAILO_REF_MIOU),
        ("Pixel accuracy (%)", a["pixel_accuracy"]), ("Mean class accuracy (%)", a["mean_class_accuracy"]),
        ("Inference latency mean ± std (ms)", f"{lat['infer_ms']['mean']} ± {lat['infer_ms']['std']}"),
        ("Inference latency p50 / p95 (ms)", f"{lat['infer_ms']['p50']} / {lat['infer_ms']['p95']}"),
        ("End-to-end latency mean ± std (ms)", f"{lat['total_ms']['mean']} ± {lat['total_ms']['std']}"),
        ("FPS end-to-end", s["throughput"]["fps_end_to_end"]), ("FPS inference-only", s["throughput"]["fps_inference_only"]),
        ("CPU utilisation mean (%)", p["cpu_total_percent"].get("mean")),
        ("SoC temperature mean / max (°C)", f"{p['soc_temp_C'].get('mean')} / {p['soc_temp_C'].get('max')}"),
        ("CPU frequency min (MHz)", p["cpu_freq_MHz"].get("min")),
        ("Process RSS mean / max (MiB)", f"{to_mib(p, 'process_rss')['mean']} / {to_mib(p, 'process_rss')['max']}"),
        ("System RAM used mean / max (MiB)", f"{to_mib(p, 'memory_used')['mean']} / {to_mib(p, 'memory_used')['max']}"),
        ("Throttling flags seen", " ".join(p["throttled_flags_seen"]) or "none"),
        ("ONNX Runtime threads", s["environment"]["intra_op_threads"]),
    ]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerows(rows)
    for k, v in rows:
        print(f"  {k:<38} {v}")


def main():
    d = sys.argv[1]
    s = json.load(open(os.path.join(d, "summary.json")))
    plots = os.path.join(d, "plots")
    os.makedirs(plots, exist_ok=True)
    per_class_iou(read_csv(os.path.join(d, "per_class_iou.csv")), s, os.path.join(plots, "per_class_iou.png"))
    latency_breakdown(s, os.path.join(plots, "latency_breakdown.png"))
    latency_per_image(read_csv(os.path.join(d, "per_image.csv")), os.path.join(plots, "latency_per_image.png"))
    platform_timeline(read_csv(os.path.join(d, "system.csv")), os.path.join(plots, "platform_timeline.png"))
    summary_table(s, os.path.join(d, "summary_table.csv"))
    print(f"Plots: {plots}\nTable: {os.path.join(d, 'summary_table.csv')}")


if __name__ == "__main__":
    main()
