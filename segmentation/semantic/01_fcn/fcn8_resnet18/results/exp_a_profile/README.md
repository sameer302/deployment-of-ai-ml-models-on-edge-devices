# Experiment A: per-layer profiling results

Produced by `../../profile_layers.py`. One folder per run: `run_<YYYYMMDD_HHMMSS>/`.

| File | Contents | In git |
|---|---|---|
| `images.txt` | The 20 val images used (every 25th sorted image) | Yes |
| `per_layer.csv` | Per layer: median / mean / std / min / max time, share of total | Yes |
| `per_layer_runs.csv` | Every layer in every profiled inference | Yes |
| `per_run.csv` | Per inference: wall time, ONNX Runtime run time, sum of layer times | Yes |
| `conditions.csv` | Per repeat: load, CPU %, temperature, clock, throttling, RAM used / available / buff-cache, process RSS | Yes |
| `summary.json` | Totals, conditions, environment | Yes |
| `ort_trace_rep<r>_*.json` | Raw ONNX Runtime profiler traces | Yes |
| `optimized_model.onnx` | The optimised graph ONNX Runtime actually runs (fused and renamed layers). Profiler layer names match this file, not the original model | **No** (~45 MB) |

## Not in git: `optimized_model.onnx`
Each run folder contains `optimized_model.onnx` on the Pi, but it is excluded by `segmentation/.gitignore`
(`**/optimized_model.onnx`) because of its size. It is fully reproducible: rerun

```bash
python profile_layers.py --repeats 1
```

with the same model file (sha256 in `../../models/README.md`) and the same ONNX Runtime version
(`summary.json` → `environment.onnxruntime`). The graph optimisations depend on the ONNX Runtime
version, so regenerate it with the version recorded in `summary.json`.
