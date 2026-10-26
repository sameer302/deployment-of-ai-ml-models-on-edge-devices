# Malloc threshold check (26 Oct 2026)

Why, with the ONNX Runtime arena off, runs 2+ peak above run 1 and keep memory between runs.
Same script as the settings experiment (`../../../ort_mem_settings.py`), arena off only, pattern on and off,
1 repeat, 6 runs, RAM sampler on (mem pass):

- `default/`: glibc 2.41 malloc with its default, self-adjusting mmap threshold
- `fixed_128KiB/`: `MALLOC_MMAP_THRESHOLD_=131072 GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072`
  (a fixed threshold also turns off the adjustment)

Each folder has the usual `child_*.json`, `timeline_*.csv`, `runs.csv`, `stages.csv`, `conditions.csv`, `summary.json`.

Result (MiB above the RAM just before run 1):

| Arena off | Threshold | Run 1 peak | Peaks runs 2–6 | Between runs | RAM before run 1 |
|---|---|---|---|---|---|
| pattern off | default | 157.5 | 172–198 | 24–54 | 221.0 |
| pattern off | fixed 128 KiB | 157.5 | 157.3–159.7 | 1–3 | 191.6 |
| pattern on | default | 157.7 | 173–175 | 24–26 | 221.0 |
| pattern on | fixed 128 KiB | 157.2 | 164–167 | 1–3 | 191.6 |

The extra memory in runs 2+ is malloc slack: after run 1 the raised threshold sends medium tensors to malloc's
heap, which keeps freed memory. With a fixed threshold every run peaks at the calculated 157.5 MiB (pattern off);
the memory pattern's planned block is about 7–9 MiB larger than the live tensors; the 29.4 MiB lower start is the
pre-processing leftovers being returned too. Latency 1–2% higher with the fixed threshold (one repeat, not conclusive).
