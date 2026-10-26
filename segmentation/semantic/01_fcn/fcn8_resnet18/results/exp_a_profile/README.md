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
| `layer_stats.csv`, `block_stats.csv`, `layer_stats.json` | Per fused layer / block: FLOPs, weight and activation sizes, live memory (`../../layer_stats.py`) | Yes |
| `analysis_layers.csv`, `analysis_blocks.csv`, `analysis.json` | Roofline and time breakdown (`../../analyze_profile.py`) | Yes |
| `plots/roofline.png`, `plots/time_vs_compute.png`, `plots/time_gap.png` | Charts from `../../analyze_profile.py` | Yes |
| `tensor_lifetimes.csv`, `arena_replay.json` | Per activation tensor: writer, last reader, size, kind; replay of the run-1/run-2 allocations through a model of ORT's arena, compared with `../arena_log_check/` (`../../tensor_lifetimes.py`) | Yes |
| `plots/tensor_lifetimes.png` | One bar per activation tensor from writer to last reader; outlined bars made the arena take a new region (`../../tensor_lifetimes.py`) | Yes |
| `plots/fusion_map.png` | The 57 ONNX layers grouped into the 30 fused layers, one column per block (`../../plot_fusion_map.py`) | Yes |
| `plots/layer_latency.png` | Median time per fused layer, coloured by roofline bound, with the best possible time (`../../plot_layer_latency.py`) | Yes |
| `plots/layer_latency_plain.png` | Median time per fused layer only, one colour (`../../plot_layer_latency.py --plain`) | Yes |
| `plots/layer_memory.png` | Weight and activation memory per fused layer, counterpart of FlexNN Fig. 1 (`../../plot_layer_memory.py`) | Yes |
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
