#!/usr/bin/env python3
"""Plot process RAM over time for each ONNX Runtime memory setting (ort_mem_settings.py results).

One panel per setting (arena x pattern), same axes: RSS sampled every 2 ms during the mem pass of one
repeat, with a thin line at the start of each inference run. Dashed line: RSS after the session
was created (weights, libraries), so everything above it is activation memory and allocator behaviour.

Output (in the run folder): plots/ram_timeline.png

Usage:
  python plot_mem_settings.py                          # latest results/ort_mem_settings/run_*
  python plot_mem_settings.py --run-dir results/ort_mem_settings/run_20261021_151457 --repeat 1
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_profile import BLUE, INK, MUTED, HERE, style

TITLES = {"arena1_pattern1": "Arena on, pattern on (default)", "arena1_pattern0": "Arena on, pattern off",
          "arena0_pattern1": "Arena off, pattern on", "arena0_pattern0": "Arena off, pattern off"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/ort_mem_settings/run_*")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/ort_mem_settings/run_*")))[-1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), dpi=150, sharex=True, sharey=True)
    for ax, (setting, title) in zip(axes.flat, TITLES.items()):
        tag = f"{setting}_mem_rep{args.repeat}"
        with open(os.path.join(run_dir, f"child_{tag}.json")) as f:
            rec = json.load(f)
        with open(os.path.join(run_dir, f"timeline_{tag}.csv")) as f:
            rows = list(csv.DictReader(f))
        t = [float(r["t_ms"]) / 1e3 for r in rows]
        rss = [float(r["rss_MiB"]) for r in rows]
        style(ax)
        ax.tick_params(labelsize=11)
        ax.grid(axis="x", visible=False)
        for run in rec["runs"]:                           # thin line at the start of every inference
            ax.axvline(run["start_ms"] / 1e3, color="#C9CEDA", linewidth=1, zorder=0)
        base = next(s["VmRSS_MiB"] for s in rec["stages"] if s["stage"] == "session_created")
        ax.axhline(base, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.plot(t, rss, color=BLUE, linewidth=1.6, zorder=2)
        peak = max(r["peak_in_run_MiB"] for r in rec["runs"])
        between = rec["runs"][-1]["rss_before_MiB"]
        ax.set_title(f"{title}: peak {peak:.0f}, between runs {between:.0f} MiB", color=INK, fontsize=12.5,
                     loc="left")
        ax.set_ylim(0, 720)
    axes[0, 0].text(0.2, base + 12, "after session created (weights + libraries)", color=MUTED, fontsize=10.5,
                    va="bottom")
    for ax in axes[1]:
        ax.set_xlabel("Time since the first run started (s)", color=INK, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Process RAM, RSS (MiB)", color=INK, fontsize=12)
    fig.suptitle("Thin vertical lines: start of each of the 10 runs", x=0.99, ha="right", fontsize=10.5, color=MUTED)
    fig.tight_layout()

    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    out = os.path.join(plots, "ram_timeline.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
