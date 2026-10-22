#!/usr/bin/env python3
"""Experiment A follow-up: effect of the ONNX Runtime memory settings on RAM and latency.

Runs FCN8-ResNet18 with each combination of the two CPU memory settings
  enable_cpu_mem_arena (arena)  x  enable_mem_pattern (pattern),   default both on,
each in a fresh Python process (memory freed by one setting must not be reused by the next),
for --repeats repeats, in two passes per setting and repeat:
  mem   RSS sampled by a background thread (RAM over time); its wake-ups slow inference (~12% in the
        quick check, 4 worker threads on 4 cores), so latency from this pass is not reported
  lat   no sampler: clean latency, comparable with the baseline run_eval.py
Peak RSS per run comes from the kernel's VmHWM counter, so it is exact in both passes. Settings are interleaved (rep 1: all four, rep 2: all four, ...) so slow drift
in temperature or background load spreads evenly. Same session settings as run_eval.py otherwise:
4 intra-op threads, 1 inter-op thread, default graph optimisation; profiler off, so latencies are
comparable with the baseline. This script measures the whole model only; RAM and time per fused
layer are a separate, later experiment.

Every child process records:
  stages     process memory after each stage: Python start, NumPy + OpenCV imported, ONNX Runtime
             imported, helpers imported (run_eval.py: onnx, psutil), raw images read, session created,
             first input ready, after run 1, after run 2, after the last run.
             VmRSS (resident), RssAnon (data: weights, tensors, arena), RssFile (library code mapped
             from disk), VmHWM (peak resident so far)
  runs       per inference: wall latency, RSS before / after, exact peak RSS during the run
             (VmHWM reset through /proc/self/clear_refs just before the run)
  timeline   process RSS read by a background thread every --sample-ms (session.run releases the
             GIL, so it keeps sampling during inference), with the start and end of every run

The images keep the same size (1024x1920), so every tensor has the same size in every run; image
content does not change memory, only system noise changes time.

Outputs (in --out-dir, default results/ort_mem_settings/run_<timestamp>/):
  child_<setting>_<pass>_rep<r>.json   raw record of one process
  timeline_<setting>_mem_rep<r>.csv    RSS samples of one mem-pass process
  stages.csv, runs.csv          all processes, long format
  conditions.csv                per process: start / end temperature, load, cool-down wait
  summary.json                  per setting: medians over repeats (latency from lat, RAM from mem)

Usage:
  python ort_mem_settings.py                       # 4 settings x 3 repeats, 10 runs each
  python ort_mem_settings.py --repeats 1 --runs 4  # quick check
  python ort_mem_settings.py --settings arena1_pattern1 --repeats 1 --runs 4   # sampler overhead test
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = {"arena1_pattern1": (True, True), "arena1_pattern0": (True, False),
            "arena0_pattern1": (False, True), "arena0_pattern0": (False, False)}
PAGE = os.sysconf("SC_PAGE_SIZE")
MIB = 2 ** 20


def proc_status():
    """VmRSS, RssAnon, RssFile, VmHWM of this process in MiB (from /proc/self/status, kB)."""
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            key = line.split(":")[0]
            if key in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                out[key + "_MiB"] = round(int(line.split()[1]) / 1024, 2)
    return out


def rss_mib():
    """Resident memory from /proc/self/statm (second field, pages): cheap enough to sample every ms."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE / MIB


def reset_peak():
    """Reset VmHWM to the current RSS (Linux: write 5 to clear_refs), so the next peak is per run."""
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


class Sampler(threading.Thread):
    """Reads RSS every `interval_s` while ONNX Runtime runs (session.run releases the GIL)."""

    def __init__(self, interval_s):
        super().__init__(daemon=True)
        self.interval_s, self.samples, self.stop_event = interval_s, [], threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.samples.append((time.monotonic_ns(), rss_mib()))
            time.sleep(self.interval_s)

    def stop(self):
        self.stop_event.set()
        self.join()


def child(args):
    """One process, one setting: memory after each stage, per-run latency and peak, RSS timeline."""
    stages = []

    def stage(name):
        stages.append({"stage": name, **proc_status()})

    stage("python_start")
    import numpy  # noqa: F401  (imported on its own line so its memory is part of this stage)
    import cv2
    stage("import_numpy_opencv")
    import onnxruntime as ort
    stage("import_onnxruntime")
    from run_eval import preprocess
    stage("import_helpers")

    raw = [cv2.imread(p) for p in args.images]
    stage("images_read")

    arena, pattern = SETTINGS[args.setting]
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = arena
    so.enable_mem_pattern = pattern
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, _, in_h, in_w = inp.shape
    stage("session_created")

    sampler = Sampler(args.sample_ms / 1e3) if args.sample_ms > 0 else None
    if sampler:
        sampler.start()
    runs = []
    for i in range(args.runs):
        x = preprocess(raw[i % len(raw)], in_w, in_h)
        if i == 0:
            stage("input_ready")
        rss_before = rss_mib()
        reset_peak()
        t0 = time.monotonic_ns()
        sess.run(None, {inp.name: x})
        t1 = time.monotonic_ns()
        st = proc_status()
        runs.append({"run": i + 1, "wall_ms": round((t1 - t0) / 1e6, 2), "t0_ns": t0, "t1_ns": t1,
                     "rss_before_MiB": round(rss_before, 2), "rss_after_MiB": st["VmRSS_MiB"],
                     "peak_in_run_MiB": st["VmHWM_MiB"]})
        if i + 1 in (1, 2) or i + 1 == args.runs:
            stage(f"after_run_{i + 1}" if i + 1 in (1, 2) else "after_last_run")
    if sampler:
        sampler.stop()

    samples = sampler.samples if sampler else []
    t_first = samples[0][0] if samples else runs[0]["t0_ns"]
    for r in runs:                                        # run windows on the timeline's clock (ms)
        r["start_ms"] = round((r.pop("t0_ns") - t_first) / 1e6, 2)
        r["end_ms"] = round((r.pop("t1_ns") - t_first) / 1e6, 2)
    record = {"setting": args.setting, "pass": args.pass_, "arena": arena, "pattern": pattern, "repeat": args.repeat,
              "threads": args.threads, "sample_ms": args.sample_ms, "page_bytes": PAGE,
              "onnxruntime": ort.__version__, "n_samples": len(samples), "stages": stages, "runs": runs}
    tag = f"{args.setting}_{args.pass_}_rep{args.repeat}"
    with open(os.path.join(args.out_dir, f"child_{tag}.json"), "w") as f:
        json.dump(record, f, indent=1)
    if not samples:
        return
    with open(os.path.join(args.out_dir, f"timeline_{tag}.csv"), "w") as f:
        f.write("t_ms,rss_MiB\n")
        f.writelines(f"{(t - t_first) / 1e6:.3f},{r:.2f}\n" for t, r in samples)


def soc_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return None


def parent(args):
    import glob
    from datetime import datetime

    import numpy as np
    from run_eval import REPO, write_csv

    out_dir = args.out_dir or os.path.join(HERE, "results", "ort_mem_settings",
                                           f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    data_root = args.data_root or os.path.join(REPO, "segmentation/datasets/cityscapes")
    images = sorted(glob.glob(os.path.join(data_root, "leftImg8bit/val/*/*_leftImg8bit.png")))[::25][: args.num_images]
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("\n".join(os.path.relpath(p, data_root) for p in images) + "\n")

    settings = [s for s in SETTINGS if s in args.settings]
    passes = args.passes
    print(f"{len(settings)} settings x {len(passes)} passes ({'/'.join(passes)}) x {args.repeats} repeats, "
          f"{args.runs} runs each, {len(images)} images, threads {args.threads}, "
          f"mem pass samples RSS every {args.sample_ms} ms -> {out_dir}", flush=True)
    conditions = []
    for rep in range(1, args.repeats + 1):
        for s, ps in ((s, ps) for s in settings for ps in passes):
            t_wait = time.monotonic()                     # cool down so every process starts alike
            while (soc_temp() or 0) > args.max_start_temp and time.monotonic() - t_wait < args.max_wait_s:
                time.sleep(5)
            cond = {"setting": s, "pass": ps, "repeat": rep, "start_temp_C": soc_temp(),
                    "load_avg_1m": round(os.getloadavg()[0], 2), "waited_s": round(time.monotonic() - t_wait, 1)}
            cmd = [sys.executable, os.path.abspath(__file__), "--child", "--setting", s, "--repeat", str(rep),
                   "--out-dir", out_dir, "--model", args.model, "--runs", str(args.runs),
                   "--threads", str(args.threads), "--pass", ps,
                   "--sample-ms", str(args.sample_ms if ps == "mem" else 0), "--images", *images]
            t0 = time.monotonic()
            subprocess.run(cmd, check=True, cwd=HERE)
            cond.update({"process_s": round(time.monotonic() - t0, 1), "end_temp_C": soc_temp()})
            conditions.append(cond)
            with open(os.path.join(out_dir, f"child_{s}_{ps}_rep{rep}.json")) as f:
                rec = json.load(f)
            walls = [r["wall_ms"] for r in rec["runs"]]
            print(f"rep {rep} {s:<16} {ps}  start {cond['start_temp_C']} C  run1 {walls[0]:.0f} ms  "
                  f"steady {np.median(walls[2:]):.0f} ms  peak {max(r['peak_in_run_MiB'] for r in rec['runs']):.0f} MiB  "
                  f"after last run {rec['stages'][-1]['VmRSS_MiB']:.0f} MiB", flush=True)

    stage_rows, run_rows = [], []
    for s in settings:
        for ps in passes:
            for rep in range(1, args.repeats + 1):
                with open(os.path.join(out_dir, f"child_{s}_{ps}_rep{rep}.json")) as f:
                    rec = json.load(f)
                key = {"setting": s, "pass": ps, "arena": int(rec["arena"]), "pattern": int(rec["pattern"]),
                       "repeat": rep}
                stage_rows += [{**key, **st} for st in rec["stages"]]
                run_rows += [{**key, **r} for r in rec["runs"]]
    write_csv(os.path.join(out_dir, "stages.csv"), stage_rows)
    write_csv(os.path.join(out_dir, "runs.csv"), run_rows)
    write_csv(os.path.join(out_dir, "conditions.csv"), conditions)

    def med(values):
        return round(float(np.median(values)), 2) if values else None

    summary = {"settings": {}, "runs_per_process": args.runs, "repeats": args.repeats, "threads": args.threads,
               "sample_ms": args.sample_ms, "passes": passes, "conditions": conditions}
    for s in settings:
        out = {}
        for ps in passes:
            rr = [r for r in run_rows if r["setting"] == s and r["pass"] == ps]
            steady = [r for r in rr if r["run"] >= 3]
            out[ps] = {
                "rss_by_stage_MiB": {name: med([r["VmRSS_MiB"] for r in stage_rows
                                                if r["setting"] == s and r["pass"] == ps and r["stage"] == name])
                                     for name in dict.fromkeys(r["stage"] for r in stage_rows)},
                "latency_run1_ms": med([r["wall_ms"] for r in rr if r["run"] == 1]),
                "latency_run2_ms": med([r["wall_ms"] for r in rr if r["run"] == 2]),
                "latency_steady_ms": med([r["wall_ms"] for r in steady]),
                "peak_in_run1_MiB": med([r["peak_in_run_MiB"] for r in rr if r["run"] == 1]),
                "peak_in_steady_run_MiB": med([r["peak_in_run_MiB"] for r in steady]),
                "rss_between_runs_MiB": med([r["rss_before_MiB"] for r in steady]),
            }
        summary["settings"][s] = out
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nmedian over repeats (steady = runs 3..N):")
    print(f"{'setting':<17}{'pass':<5}{'run1 ms':>9}{'steady ms':>10}{'peak run1':>10}{'peak steady':>12}{'between':>9}")
    for s, by_pass in summary["settings"].items():
        for ps, v in by_pass.items():
            print(f"{s:<17}{ps:<5}{v['latency_run1_ms']:>9}{v['latency_steady_ms']:>10}{v['peak_in_run1_MiB']:>10}"
                  f"{v['peak_in_steady_run_MiB']:>12}{v['rss_between_runs_MiB']:>9}")
    print(f"Results: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    ap.add_argument("--data-root", default=None, help="default: segmentation/datasets/cityscapes")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--repeats", type=int, default=3, help="fresh process per setting and repeat")
    ap.add_argument("--runs", type=int, default=10, help="inferences per process (no warm-up: runs 1 and 2 matter)")
    ap.add_argument("--num-images", type=int, default=5, help="every 25th val image, cycled over the runs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--passes", nargs="+", default=["mem", "lat"], choices=["mem", "lat"],
                    help="mem: RSS sampler on (RAM over time); lat: sampler off (clean latency)")
    ap.add_argument("--sample-ms", type=float, default=2.0, help="RSS sampling interval in the mem pass")
    ap.add_argument("--max-start-temp", type=float, default=55.0, help="wait until the SoC is below this (C)")
    ap.add_argument("--max-wait-s", type=float, default=180.0)
    # internal: one child process
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--setting", choices=list(SETTINGS), help=argparse.SUPPRESS)
    ap.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--pass", dest="pass_", choices=["mem", "lat"], help=argparse.SUPPRESS)
    ap.add_argument("--images", nargs="+", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
