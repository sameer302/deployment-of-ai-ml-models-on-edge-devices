#!/usr/bin/env python3
"""Follow-up to the ORT memory settings experiment: what the memory arena does in runs 1 and 2.

The settings experiment (ort_mem_settings.py) measured, with the arena on, that run 1 raises RAM by
~324 MiB for a 157.5 MiB activation peak, and that run 2 adds a further ~56 MiB with the memory pattern
on and ~105 MiB with it off. This script checks the explanation, once, from ONNX Runtime's own arena log:
  - the arena (BFCArena) grows in regions, each at least as large as the request and, by default,
    doubling in size (arena_extend_strategy kNextPowerOfTwo)
  - neighbouring free pieces in one region are merged; pieces in different regions are not
  - RSS only counts pages that have been written, so reserving a region is not the same as using it

With the default logger at INFO, ONNX Runtime prints every arena event:
  "Extending BFCArena for Cpu. bin_num:.. (requested) num_bytes: N (actual) rounded_bytes:R"
  "Extended allocation by E bytes."          region taken from the OS
  "Total allocated bytes: T"                 everything the arena holds now
  "Reserving memory in BFCArena for Cpu size: S"   a block allocated outside the arena's regions
The child process writes a marker line with its RSS to the same stream before and after every run, so
each event can be assigned to "session creation", "run 1", "run 2", ...

Per setting (arena on, pattern on / off; fresh process each), for each segment: number of extensions,
MiB requested, MiB taken from the OS (region sizes), reservations, arena total at the end, RSS before
and after, and the exact peak RSS (VmHWM reset before each run). Same model, threads and pre-processing
as ort_mem_settings.py; one image (tensor sizes do not depend on the image).

Outputs (in --out-dir, default results/arena_log_check/run_<timestamp>/):
  log_<setting>.txt     raw stderr of the child: ORT log + markers
  events.csv            one row per arena event, with its segment
  segments.csv          one row per setting and segment: arena and RSS numbers side by side
  summary.json          the same, plus versions and settings

Usage:
  python arena_log_check.py               # arena on, pattern on and off, 4 runs each
  python arena_log_check.py --runs 3
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = {"arena1_pattern1": (True, True), "arena1_pattern0": (True, False)}
MIB = 2 ** 20
MARK = "@@MARK"

RE_EXTEND = re.compile(r"Extending BFCArena for \S+?\. bin_num:(-?\d+) \(requested\) num_bytes: (\d+) \(actual\) rounded_bytes:(\d+)")
RE_EXTENDED = re.compile(r"Extended allocation by (\d+) bytes")
RE_TOTAL = re.compile(r"Total allocated bytes: (\d+)")
RE_RESERVE = re.compile(r"Reserving memory in BFCArena for \S+ size: (\d+)")
RE_MARK = re.compile(MARK + r" (\S+) rss=([\d.]+) hwm=([\d.]+)")


def proc_mib(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
    return float("nan")


def mark(name):
    """Marker on stderr (fd 2, unbuffered, same stream as ORT's log) with RSS and peak RSS in MiB."""
    os.write(2, f"{MARK} {name} rss={proc_mib('VmRSS'):.2f} hwm={proc_mib('VmHWM'):.2f}\n".encode())


def child(args):
    import cv2
    import onnxruntime as ort
    from run_eval import preprocess

    ort.set_default_logger_severity(args.log_severity)   # 1 = INFO: arena events are logged at INFO
    raw = cv2.imread(args.image)
    arena, pattern = SETTINGS[args.setting]
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = arena
    so.enable_mem_pattern = pattern
    mark("start")
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, _, in_h, in_w = inp.shape
    x = preprocess(raw, in_w, in_h)
    mark("session_and_input_ready")
    for i in range(1, args.runs + 1):
        with open("/proc/self/clear_refs", "w") as f:   # reset VmHWM: next peak is this run's
            f.write("5")
        mark(f"run{i}_start")
        sess.run(None, {inp.name: x})
        mark(f"run{i}_end")


def parse(log_path, setting):
    """Assign every arena event to the segment it happened in (session, run1, run2, ...)."""
    events, marks, seg = [], {}, "session"
    with open(log_path, errors="replace") as f:
        for line in f:
            m = RE_MARK.search(line)
            if m:
                name, rss, hwm = m.group(1), float(m.group(2)), float(m.group(3))
                marks[name] = {"rss": rss, "hwm": hwm}
                if name.endswith("_start") and name.startswith("run"):
                    seg = name[:-6]
                continue
            for rx, kind in ((RE_EXTEND, "extend_request"), (RE_EXTENDED, "extended"),
                             (RE_TOTAL, "total"), (RE_RESERVE, "reserve")):
                m = rx.search(line)
                if not m:
                    continue
                row = {"setting": setting, "segment": seg, "event": kind,
                       "requested_MiB": "", "rounded_MiB": "", "MiB": ""}
                if kind == "extend_request":
                    row.update(requested_MiB=round(int(m.group(2)) / MIB, 3), rounded_MiB=round(int(m.group(3)) / MIB, 3))
                else:
                    row["MiB"] = round(int(m.group(1)) / MIB, 3)
                events.append(row)
                break
    return events, marks


def summarise(events, marks, setting, runs):
    rows = []
    for seg in ["session"] + [f"run{i}" for i in range(1, runs + 1)]:
        ev = [e for e in events if e["segment"] == seg]
        totals = [e["MiB"] for e in ev if e["event"] == "total"]
        row = {"setting": setting, "segment": seg,
               "n_extensions": sum(e["event"] == "extended" for e in ev),
               "requested_MiB": round(sum(e["requested_MiB"] or 0 for e in ev), 2),
               "taken_from_os_MiB": round(sum(e["MiB"] for e in ev if e["event"] == "extended"), 2),
               "regions_MiB": " ".join(f"{e['MiB']:.0f}" for e in ev if e["event"] == "extended"),
               "reserved_MiB": round(sum(e["MiB"] for e in ev if e["event"] == "reserve"), 2),
               "arena_total_end_MiB": totals[-1] if totals else ""}
        if seg == "session":
            before, after = marks.get("start"), marks.get("session_and_input_ready")
        else:
            before, after = marks.get(f"{seg}_start"), marks.get(f"{seg}_end")
        if before and after:
            row.update(rss_before_MiB=before["rss"], rss_after_MiB=after["rss"],
                       rss_change_MiB=round(after["rss"] - before["rss"], 2),
                       peak_rss_MiB=after["hwm"] if seg != "session" else "")
        rows.append(row)
    return rows


def parent(args):
    import glob
    from datetime import datetime

    import onnxruntime as ort
    from run_eval import REPO, write_csv

    out_dir = args.out_dir or os.path.join(HERE, "results", "arena_log_check",
                                           f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    data_root = args.data_root or os.path.join(REPO, "segmentation/datasets/cityscapes")
    image = sorted(glob.glob(os.path.join(data_root, "leftImg8bit/val/*/*_leftImg8bit.png")))[0]

    all_events, all_rows = [], []
    for setting in SETTINGS:
        log_path = os.path.join(out_dir, f"log_{setting}.txt")
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--setting", setting, "--image", image,
               "--model", args.model, "--runs", str(args.runs), "--threads", str(args.threads),
               "--log-severity", str(args.log_severity)]
        print(f"{setting}: running {args.runs} inferences ...", flush=True)
        with open(log_path, "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=HERE)
        events, marks = parse(log_path, setting)
        if not any(e["event"] == "extended" for e in events):
            print(f"  warning: no arena events in {log_path}; check the log severity")
        all_events += events
        all_rows += summarise(events, marks, setting, args.runs)
        time.sleep(args.pause_s)

    if all_events:
        write_csv(os.path.join(out_dir, "events.csv"), all_events)
    write_csv(os.path.join(out_dir, "segments.csv"), all_rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"onnxruntime": ort.__version__, "threads": args.threads, "runs": args.runs,
                   "image": os.path.relpath(image, data_root), "segments": all_rows}, f, indent=1)

    cols = ["segment", "n_extensions", "requested_MiB", "taken_from_os_MiB", "reserved_MiB",
            "arena_total_end_MiB", "rss_change_MiB", "peak_rss_MiB"]
    for setting in SETTINGS:
        print(f"\n{setting}")
        print("  " + " | ".join(cols))
        for r in all_rows:
            if r["setting"] == setting:
                print("  " + " | ".join(str(r.get(c, "")) for c in cols) + f"   regions: {r['regions_MiB']}")
    print("\nwrote", out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    ap.add_argument("--data-root", default=None, help="default: segmentation/datasets/cityscapes")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log-severity", type=int, default=1, help="ORT default logger: 0 verbose, 1 info")
    ap.add_argument("--pause-s", type=float, default=20.0, help="cool-down between the two processes")
    # internal: one child process
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--setting", choices=list(SETTINGS), help=argparse.SUPPRESS)
    ap.add_argument("--image", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
