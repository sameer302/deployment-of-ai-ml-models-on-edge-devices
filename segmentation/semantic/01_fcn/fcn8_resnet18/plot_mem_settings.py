#!/usr/bin/env python3
"""Plot process RAM over time for each ONNX Runtime memory setting (ort_mem_settings.py results).

One panel per setting (arena x pattern), same axes: RSS sampled every 2 ms during the mem pass of one
repeat, with a thin line at the start of each inference run. By default RAM is shown above the level just
before run 1 (input ready: fixed overhead + weights + input, ~222 MiB in every setting), so the curve is
what inference adds. Reference lines: the calculated activation peak (157.5 MiB, layer_stats.json) and,
in the arena-on panels, what the arena has reserved after run 1 (latest results/arena_log_check run).
--total plots total process RAM instead.

Output (in the run folder): plots/ram_timeline.png

Usage:
  python plot_mem_settings.py                          # latest results/ort_mem_settings/run_*
  python plot_mem_settings.py --run-dir results/ort_mem_settings/run_20261021_151457 --repeat 1
  python plot_mem_settings.py --total                  # total process RAM
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze_profile import BLUE, INK, MUTED, ORANGE, HERE, style

TITLES = {"arena1_pattern1": "Arena on, pattern on (default)", "arena1_pattern0": "Arena on, pattern off",
          "arena0_pattern1": "Arena off, pattern on", "arena0_pattern0": "Arena off, pattern off"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="default: latest results/ort_mem_settings/run_*")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--total", action="store_true", help="total process RAM instead of RAM above run-1 start")
    args = ap.parse_args()
    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/ort_mem_settings/run_*")))[-1]

    act_peak = arena_reserved = None
    stats = sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*/layer_stats.json")))
    if stats:
        with open(stats[-1]) as f:
            act_peak = json.load(f)["activation_peak"]["live_act_MiB"]
    logs = sorted(glob.glob(os.path.join(HERE, "results/arena_log_check/run_*/segments.csv")))
    if logs:
        with open(logs[-1]) as f:
            arena_reserved = next((float(r["arena_total_end_MiB"]) for r in csv.DictReader(f)
                                   if r["segment"] == "run1" and r["arena_total_end_MiB"]), None)

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), dpi=150, sharex=True, sharey=True)
    for ax, (setting, title) in zip(axes.flat, TITLES.items()):
        tag = f"{setting}_mem_rep{args.repeat}"
        with open(os.path.join(run_dir, f"child_{tag}.json")) as f:
            rec = json.load(f)
        with open(os.path.join(run_dir, f"timeline_{tag}.csv")) as f:
            rows = list(csv.DictReader(f))
        t = [float(r["t_ms"]) / 1e3 for r in rows]
        base = 0 if args.total else next(s["VmRSS_MiB"] for s in rec["stages"] if s["stage"] == "input_ready")
        rss = [float(r["rss_MiB"]) - base for r in rows]
        style(ax)
        ax.tick_params(labelsize=11)
        ax.grid(axis="x", visible=False)
        for run in rec["runs"]:                           # thin line at the start of every inference
            ax.axvline(run["start_ms"] / 1e3, color="#C9CEDA", linewidth=1, zorder=0)
        if args.total:
            session = next(s["VmRSS_MiB"] for s in rec["stages"] if s["stage"] == "session_created")
            ax.axhline(session, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            ax.text(0.2, session + 12, "after session created (weights + libraries)", color=MUTED, fontsize=10.5,
                    va="bottom")
        else:
            if act_peak:
                ax.axhline(act_peak, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            if arena_reserved and rec["arena"]:
                ax.axhline(arena_reserved, color=ORANGE, linewidth=1.4, linestyle=(0, (2, 2)), zorder=1)
        ax.plot(t, rss, color=BLUE, linewidth=1.6, zorder=2)
        peak = max(r["peak_in_run_MiB"] for r in rec["runs"]) - base
        between = rec["runs"][-1]["rss_before_MiB"] - base
        ax.set_title(f"{title}: peak {peak:.0f}, between runs {between:.0f} MiB", color=INK, fontsize=12.5,
                     loc="left")
        ax.set_ylim(0, 720 if args.total else 580)
    for ax in axes[1]:
        ax.set_xlabel("Time since the first run started (s)", color=INK, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Process RAM, RSS (MiB)" if args.total else "RAM above start of run 1 (MiB)",
                      color=INK, fontsize=12)
    fig.suptitle("Thin vertical lines: start of each of the 10 runs", x=0.99, ha="right", fontsize=10.5, color=MUTED)
    fig.tight_layout()
    if not args.total:
        handles = []
        if act_peak:
            handles.append(Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                                  label=f"Activations, calculated peak ({act_peak:.1f} MiB)"))
        if arena_reserved:
            handles.append(Line2D([], [], color=ORANGE, linewidth=1.4, linestyle=(0, (2, 2)),
                                  label=f"Arena reserved after run 1 ({arena_reserved:.0f} MiB, arena on)"))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.985), ncol=2, frameon=False,
                   fontsize=10.5)

    plots = os.path.join(run_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    out = os.path.join(plots, "ram_timeline_total.png" if args.total else "ram_timeline.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
