#!/usr/bin/env python3
"""Experiment A, script 1: time of every layer of FCN8-ResNet18 with the ONNX Runtime profiler.

Same input pipeline and session settings as run_eval.py (4 intra-op threads, 1 inter-op thread,
default graph optimisation). Every 25th Cityscapes val image (20 images, all three cities) is read
into memory first (raw, ~6 MiB each) and preprocessed just before its inference, outside the timed
window, so only inference is timed. Each repeat uses a fresh session, logs system conditions and
RAM, runs warm-up inferences, then profiles one inference per image.

Outputs (in --out-dir):
  images.txt                  the images used
  optimized_model.onnx        the graph ONNX Runtime actually runs (fused / renamed layers);
                              profiler layer names match this file, not the original model
  ort_trace_rep<r>_*.json     raw ONNX Runtime profiler traces
  per_layer_runs.csv          one row per layer per profiled inference
  per_layer.csv               per layer: median / mean / std / min / max time over all runs
  per_run.csv                 per inference: wall time, ORT run time, sum of layer times
  conditions.csv              per repeat: load, CPU %, temperature, clock, throttling, RAM
  summary.json                conditions, totals, environment

Usage:
  python profile_layers.py                      # 20 images, 3 repeats
  python profile_layers.py --repeats 1 --stride 50
"""
import argparse
import glob
import json
import os
import resource
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from run_eval import HERE, MIB, REPO, environment_info, preprocess, read_sysfs, read_throttled, write_csv


def conditions():
    """System state just before a repeat starts (after its session is created). RAM in MiB."""
    freqs = [read_sysfs(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq", 1000)
             for c in range(os.cpu_count())]
    vm = psutil.virtual_memory()
    return {
        "timestamp": datetime.now().isoformat(),
        "load_avg_1m": round(os.getloadavg()[0], 2),
        "cpu_percent_1s": psutil.cpu_percent(interval=1.0),
        "soc_temp_C": read_sysfs("/sys/class/thermal/thermal_zone0/temp", 1000),
        "cpu_freq_MHz": "/".join(str(round(f)) for f in freqs if f is not None),
        "throttled": read_throttled(),
        "ram_used_MiB": round(vm.used / MIB, 1),
        "ram_available_MiB": round(vm.available / MIB, 1),
        "ram_buff_cache_MiB": round((vm.buffers + vm.cached) / MIB, 1),
        "process_rss_MiB": round(psutil.Process().memory_info().rss / MIB, 1),
    }


def peak_rss_mib():
    """Highest RSS of this process so far (Linux reports ru_maxrss in KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def parse_trace(path, warmup):
    """Split a profiler trace into inferences and return per-layer times of the profiled ones.

    The trace has one Session event "model_run" per inference and one Node event
    "<layer>_kernel_time" per layer execution; layers are assigned to the run whose
    time window contains them. The first `warmup` runs are dropped.
    """
    with open(path) as f:
        events = json.load(f)
    runs = sorted((e for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"),
                  key=lambda e: e["ts"])
    nodes = sorted((e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")),
                   key=lambda e: e["ts"])
    out = []
    for r in runs[warmup:]:
        start, end = r["ts"], r["ts"] + r["dur"]
        layers = [{"layer": n["name"][: -len("_kernel_time")],
                   "op_type": n.get("args", {}).get("op_name", ""),
                   "ts_us": n["ts"] - start,
                   "time_ms": n["dur"] / 1e3}
                  for n in nodes if start <= n["ts"] <= end]
        out.append({"run_ms": r["dur"] / 1e3, "layers": layers})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "segmentation/datasets/cityscapes"))
    ap.add_argument("--out-dir", default=None, help="default: results/exp_a_profile/run_<timestamp>")
    ap.add_argument("--stride", type=int, default=25, help="take every N-th sorted val image")
    ap.add_argument("--num-images", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3, help="fresh session per repeat")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up inferences per repeat, not counted")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(HERE, "results", "exp_a_profile", f"run_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    all_images = sorted(glob.glob(os.path.join(args.data_root, "leftImg8bit/val/*/*_leftImg8bit.png")))
    images = all_images[:: args.stride][: args.num_images]
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("\n".join(os.path.relpath(p, args.data_root) for p in images) + "\n")

    # Read raw images first so disk reads stay out of the loop; preprocess per image (not timed),
    # which keeps process RAM close to run_eval.py (20 preprocessed inputs would add ~450 MiB).
    probe = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = probe.get_inputs()[0]
    _, _, in_h, in_w = inp.shape
    del probe
    raw = [cv2.imread(p) for p in images]
    print(f"{len(images)} images (every {args.stride}th), input {in_w}x{in_h}, "
          f"{args.repeats} repeats x ({args.warmup} warm-up + {len(images)} profiled), threads {args.threads}")

    run_rows, layer_rows, cond_rows = [], [], []
    env = None
    for rep in range(1, args.repeats + 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(out_dir, f"ort_trace_rep{rep}")
        if rep == 1:
            so.optimized_model_filepath = os.path.join(out_dir, "optimized_model.onnx")
        sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
        if env is None:
            env = environment_info(args, sess)

        cond = {"repeat": rep, **conditions()}
        print(f"repeat {rep}: load {cond['load_avg_1m']}  cpu {cond['cpu_percent_1s']}%  "
              f"temp {cond['soc_temp_C']} C  throttled {cond['throttled']}  "
              f"RAM used {cond['ram_used_MiB']} MiB  process RSS {cond['process_rss_MiB']} MiB", flush=True)

        first = preprocess(raw[0], in_w, in_h)
        for _ in range(args.warmup):
            sess.run(None, {inp.name: first})
        walls, rss_after = [], []
        for bgr in raw:
            x = preprocess(bgr, in_w, in_h)
            t0 = time.perf_counter()
            sess.run(None, {inp.name: x})
            walls.append(1e3 * (time.perf_counter() - t0))
            rss_after.append(psutil.Process().memory_info().rss / MIB)
        trace = sess.end_profiling()
        cond.update({"process_rss_after_run_max_MiB": round(max(rss_after), 1),
                     "process_peak_rss_so_far_MiB": peak_rss_mib(),
                     "ram_used_end_MiB": round(psutil.virtual_memory().used / MIB, 1)})
        cond_rows.append(cond)

        runs = parse_trace(trace, args.warmup)
        if len(runs) != len(raw):
            raise RuntimeError(f"{trace}: expected {len(raw)} profiled runs, found {len(runs)}")
        for i, (r, wall) in enumerate(zip(runs, walls)):
            run_rows.append({"repeat": rep, "run": i, "image": os.path.basename(images[i]).replace("_leftImg8bit.png", ""),
                             "wall_ms": round(wall, 2), "ort_run_ms": round(r["run_ms"], 2),
                             "sum_layers_ms": round(sum(l["time_ms"] for l in r["layers"]), 2),
                             "num_layers": len(r["layers"])})
            for order, l in enumerate(r["layers"]):
                layer_rows.append({"repeat": rep, "run": i, "order": order, "layer": l["layer"],
                                   "op_type": l["op_type"], "start_us": l["ts_us"], "time_ms": round(l["time_ms"], 4)})
        print(f"  median wall {np.median(walls):.0f} ms, median ORT run {np.median([r['run_ms'] for r in runs]):.0f} ms, "
              f"{len(runs[0]['layers'])} layers, peak RSS so far {cond['process_peak_rss_so_far_MiB']} MiB", flush=True)
        del sess

    # Per layer over all profiled runs; layers keep their execution order.
    times, op_of, order_of = defaultdict(list), {}, {}
    for row in layer_rows:
        times[row["layer"]].append(row["time_ms"])
        op_of[row["layer"]] = row["op_type"]
        order_of.setdefault(row["layer"], row["order"])
    total_median = sum(np.median(v) for v in times.values())
    per_layer = []
    for name in sorted(times, key=order_of.get):
        a = np.asarray(times[name])
        per_layer.append({"order": order_of[name], "layer": name, "op_type": op_of[name], "n": a.size,
                          "median_ms": round(float(np.median(a)), 4), "mean_ms": round(float(a.mean()), 4),
                          "std_ms": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
                          "min_ms": round(float(a.min()), 4), "max_ms": round(float(a.max()), 4),
                          "share_pct": round(100 * float(np.median(a)) / total_median, 3)})

    write_csv(os.path.join(out_dir, "per_layer_runs.csv"), layer_rows)
    write_csv(os.path.join(out_dir, "per_layer.csv"), per_layer)
    write_csv(os.path.join(out_dir, "per_run.csv"), run_rows)
    write_csv(os.path.join(out_dir, "conditions.csv"), cond_rows)

    med = lambda k: round(float(np.median([r[k] for r in run_rows])), 2)
    summary = {
        "experiment": "A - per-layer profiling", "model": os.path.basename(args.model),
        "runtime": "onnxruntime", "precision": "FP32", "input_size": [in_h, in_w],
        "images": len(images), "stride": args.stride, "repeats": args.repeats, "warmup": args.warmup,
        "num_layers_profiled": len(per_layer),
        "median_ms": {"wall": med("wall_ms"), "ort_run": med("ort_run_ms"), "sum_layers": med("sum_layers_ms")},
        "sum_of_layer_medians_ms": round(total_median, 2),
        "process_peak_rss_MiB": peak_rss_mib(),
        "per_repeat_median_wall_ms": [round(float(np.median([r["wall_ms"] for r in run_rows if r["repeat"] == k])), 2)
                                      for k in range(1, args.repeats + 1)],
        "conditions": cond_rows,
        "environment": env,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmedian per inference: wall {summary['median_ms']['wall']} ms | ORT run {summary['median_ms']['ort_run']} ms | "
          f"sum of layers {summary['median_ms']['sum_layers']} ms | {len(per_layer)} layers")
    print("top 5 layers by median time:")
    for l in sorted(per_layer, key=lambda l: -l["median_ms"])[:5]:
        print(f"  {l['layer']:<40} {l['op_type']:<12} {l['median_ms']:8.2f} ms  {l['share_pct']:5.1f}%")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
