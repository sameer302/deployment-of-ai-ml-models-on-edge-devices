"""FCN8-ResNet18 (ONNX, FP32) on Cityscapes val: model accuracy + platform metrics on the Pi 5 CPU.

Writes to --out-dir:
  per_image.csv      one row per image: ISO timestamp, per-stage latency (load/pre/infer/post), pixel accuracy
  system.csv         background samples every --interval s: CPU % (total + per core), freq, SoC temp, throttling,
                     RAM used, page cache (buff/cache), process RSS - memory in MiB (2^20 bytes, as `free -m`)
  per_class_iou.csv  IoU for each of the 19 Cityscapes classes
  summary.json       mIoU, pixel accuracy, latency stats (mean/std/p50/p95), FPS, platform stats, environment

Usage (from repo root):
  venv_cpu/bin/python segmentation/semantic/01_fcn/fcn8_resnet18/run_eval.py --num-images 500 --threads 4
"""
import argparse
import csv
import glob
import json
import os
import platform
import signal
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]
NUM_CLASSES = len(CLASSES)
IGNORE = 255
# Cityscapes labelId -> trainId; every other labelId is ignored in evaluation.
LABEL_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                  24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
# MMSegmentation ImageNet normalization, RGB order (confirmed by the sanity check: RGB beats BGR).
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
MIB = 2 ** 20  # memory is logged in MiB, the unit `free -m` reports


def read_sysfs(path, scale):
    try:
        with open(path) as f:
            return int(f.read().strip()) / scale
    except Exception:
        return None


def read_throttled():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip().split("=")[1]
    except Exception:
        return None


class SystemMonitor(threading.Thread):
    """Samples platform metrics every `interval` seconds into an in-memory list (flushed at the end)."""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.rows = []
        self.stop_event = threading.Event()
        self.proc = psutil.Process()

    def run(self):
        psutil.cpu_percent(percpu=True)  # prime: first call has no previous window
        self.proc.cpu_percent()
        t0 = time.monotonic()
        while not self.stop_event.wait(self.interval):
            per_core = psutil.cpu_percent(percpu=True)
            vm = psutil.virtual_memory()
            self.rows.append({
                "timestamp": datetime.now().isoformat(),
                "elapsed_s": round(time.monotonic() - t0, 3),
                "cpu_total_percent": round(sum(per_core) / len(per_core), 1),
                **{f"cpu{i}_percent": v for i, v in enumerate(per_core)},
                "process_cpu_percent": self.proc.cpu_percent(),
                "cpu_freq_MHz": read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000),
                "soc_temp_C": read_sysfs("/sys/class/thermal/thermal_zone0/temp", 1000),
                "throttled_flags_hex": read_throttled(),
                "memory_percent": vm.percent,
                "memory_used_MiB": round(vm.used / MIB, 1),
                "memory_buff_cache_MiB": round((vm.buffers + vm.cached) / MIB, 1),
                "process_rss_MiB": round(self.proc.memory_info().rss / MIB, 1),
            })

    def stop(self):
        self.stop_event.set()
        self.join()


def preprocess(bgr, width, height):
    x = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    x = (x[:, :, ::-1].astype(np.float32) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def environment_info(args, sess):
    governors = sorted({open(p).read().strip() for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")})
    return {
        "timestamp": datetime.now().isoformat(),
        "os": platform.freedesktop_os_release().get("PRETTY_NAME", "unknown"),
        "kernel": os.uname().release,
        "python": platform.python_version(),
        "numpy": np.__version__, "opencv": cv2.__version__, "onnxruntime": ort.__version__,
        "providers": sess.get_providers(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "governors": governors,
        "intra_op_threads": args.threads,
        "start_soc_temp_C": read_sysfs("/sys/class/thermal/thermal_zone0/temp", 1000),
        "start_throttled": read_throttled(),
        "memory_total_MiB": round(psutil.virtual_memory().total / MIB, 1),
        "start_memory_used_MiB": round(psutil.virtual_memory().used / MIB, 1),
        "start_memory_buff_cache_MiB": round((psutil.virtual_memory().buffers + psutil.virtual_memory().cached) / MIB, 1),
    }


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    return {"mean": round(a.mean(), 2), "std": round(a.std(ddof=1) if a.size > 1 else 0.0, 2),
            "p50": round(np.percentile(a, 50), 2), "p95": round(np.percentile(a, 95), 2),
            "min": round(a.min(), 2), "max": round(a.max(), 2)}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "segmentation/datasets/cityscapes"))
    ap.add_argument("--out-dir", default=None, help="default: results/ort_fp32_t<threads>_<timestamp>")
    ap.add_argument("--num-images", type=int, default=0, help="0 = all 500 val images")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up inferences, excluded from results")
    ap.add_argument("--interval", type=float, default=1.0, help="system sampling interval (s)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(HERE, "results", f"ort_fp32_t{args.threads}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.data_root, "leftImg8bit/val/*/*_leftImg8bit.png")))
    if args.num_images:
        images = images[: args.num_images]
    lut = np.full(256, IGNORE, np.uint8)
    for lid, tid in LABEL_TO_TRAIN.items():
        lut[lid] = tid

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, _, in_h, in_w = inp.shape
    params = sum(int(np.prod(t.dims)) for t in onnx.load(args.model).graph.initializer)
    env = environment_info(args, sess)
    print(f"Model {os.path.basename(args.model)} input {in_w}x{in_h}, {params/1e6:.2f} M params | "
          f"{len(images)} images | threads {args.threads} | governor {env['governors']} | "
          f"start temp {env['start_soc_temp_C']} C")

    # Stop cleanly on Ctrl+C / kill: results gathered so far are still written.
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    first = preprocess(cv2.imread(images[0]), in_w, in_h)
    for _ in range(args.warmup):
        sess.run(None, {inp.name: first})

    monitor = SystemMonitor(args.interval)
    monitor.start()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    rows = []
    t_run = time.perf_counter()
    try:
        for i, path in enumerate(images):
            if stop["flag"]:
                break
            label_path = path.replace("leftImg8bit/val", "gtFine/val").replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            t0 = time.perf_counter()
            bgr = cv2.imread(path)
            gt = lut[cv2.imread(label_path, cv2.IMREAD_UNCHANGED)]
            t1 = time.perf_counter()
            x = preprocess(bgr, in_w, in_h)
            t2 = time.perf_counter()
            out = sess.run(None, {inp.name: x})[0]
            t3 = time.perf_counter()
            pred = cv2.resize(out[0, 0].astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            t4 = time.perf_counter()

            valid = gt != IGNORE
            conf += np.bincount(NUM_CLASSES * gt[valid].astype(np.int64) + pred[valid],
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            rows.append({
                "timestamp": datetime.now().isoformat(),
                "image": os.path.basename(path).replace("_leftImg8bit.png", ""),
                "load_ms": round(1e3 * (t1 - t0), 2),
                "pre_ms": round(1e3 * (t2 - t1), 2),
                "infer_ms": round(1e3 * (t3 - t2), 2),
                "post_ms": round(1e3 * (t4 - t3), 2),
                "total_ms": round(1e3 * (t4 - t0), 2),
                "pixel_acc": round(float((pred[valid] == gt[valid]).mean()), 4),
            })
            if (i + 1) % 25 == 0 or i + 1 == len(images):
                done = time.perf_counter() - t_run
                print(f"  {i+1}/{len(images)}  last infer {rows[-1]['infer_ms']:.0f} ms  "
                      f"elapsed {done/60:.1f} min  ETA {(len(images)-i-1)*done/(i+1)/60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("Interrupted - writing partial results")
    wall_s = time.perf_counter() - t_run
    monitor.stop()

    # Accuracy from the accumulated confusion matrix (rows = ground truth, cols = prediction).
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    class_acc = np.where(conf.sum(1) > 0, tp / np.maximum(conf.sum(1), 1), np.nan)

    write_csv(os.path.join(out_dir, "per_image.csv"), rows)
    write_csv(os.path.join(out_dir, "system.csv"), monitor.rows)
    write_csv(os.path.join(out_dir, "per_class_iou.csv"),
              [{"class_id": c, "class": CLASSES[c], "iou": None if np.isnan(iou[c]) else round(float(iou[c]), 4),
                "gt_pixels": int(conf[c].sum())} for c in range(NUM_CLASSES)])
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), conf)

    sysr = monitor.rows
    summary = {
        "model": os.path.basename(args.model), "runtime": "onnxruntime", "precision": "FP32",
        "input_size": [in_h, in_w], "params_M": round(params / 1e6, 2),
        "dataset": "Cityscapes val", "images_evaluated": len(rows),
        "accuracy": {
            "mIoU": round(float(np.nanmean(iou)) * 100, 2),
            "pixel_accuracy": round(float(tp.sum() / max(conf.sum(), 1)) * 100, 2),
            "mean_class_accuracy": round(float(np.nanmean(class_acc)) * 100, 2),
        },
        "latency_ms": {k: stats([r[k] for r in rows]) for k in ("load_ms", "pre_ms", "infer_ms", "post_ms", "total_ms")},
        "throughput": {
            "fps_end_to_end": round(len(rows) / wall_s, 3) if rows else None,
            "fps_inference_only": round(1e3 / np.mean([r["infer_ms"] for r in rows]), 3) if rows else None,
            "wall_time_s": round(wall_s, 1),
        },
        "platform": {
            "cpu_total_percent": stats([r["cpu_total_percent"] for r in sysr]),
            "soc_temp_C": stats([r["soc_temp_C"] for r in sysr if r["soc_temp_C"] is not None]),
            "cpu_freq_MHz": stats([r["cpu_freq_MHz"] for r in sysr if r["cpu_freq_MHz"] is not None]),
            "process_rss_MiB": stats([r["process_rss_MiB"] for r in sysr]),
            "memory_used_MiB": stats([r["memory_used_MiB"] for r in sysr]),
            "memory_buff_cache_MiB": stats([r["memory_buff_cache_MiB"] for r in sysr]),
            "throttled_flags_seen": sorted({r["throttled_flags_hex"] for r in sysr if r["throttled_flags_hex"]}),
        },
        "environment": env,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmIoU {summary['accuracy']['mIoU']}% | pixel acc {summary['accuracy']['pixel_accuracy']}% | "
          f"infer {summary['latency_ms']['infer_ms'].get('mean')} ms | e2e FPS {summary['throughput']['fps_end_to_end']} | "
          f"max temp {summary['platform']['soc_temp_C'].get('max')} C")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
