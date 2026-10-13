# Pending items

Things deferred until first results are in. Tick off as they are done.

## Understanding
- [ ] FCN8-ResNet18 architecture in detail (encoder, FCN-8s decoder, pool3/pool4 skips, upsampling)
- [ ] Cityscapes dataset in detail (class definitions, labelIds vs trainIds, ignore label, splits)

## Experiments
- [x] Full 500-image Cityscapes val run, ORT FP32, 4 threads (2026-10-10): mIoU 69.87%, 1,672 ms inference, 0.54 FPS e2e -> `results/ort_fp32_t4_full/`
- [ ] Repeat runs (>= 3) for mean ± std across runs, without Claude Code running (guidelines §2, §4b)

### CPU runtime x precision sequence (change one dimension per step)
| # | Runtime | Precision | Compare with | Status |
|---|---|---|---|---|
| 1 | PyTorch | FP32 | #2 (framework effect) | Optional, needs `venv_export` + original checkpoint |
| 2 | ONNX Runtime | FP32 | Published mIoU (pipeline check) | Done (69.87% vs 69.4% ref) |
| 3 | NCNN | FP32 (fp16 disabled) | #2 (runtime effect) | Pending (next) |
| 4 | NCNN | FP16 (A76 native) | #3 (precision effect) | Pending |
| 5 | ONNX Runtime | INT8 static | #2 (quantization effect) | Pending |
| 6 | NCNN | INT8 | #3 (quantization effect) | Pending |

- NCNN enables fp16 by default on Cortex-A76: disable `use_fp16_*` options for the true FP32 run (#3).
- Discuss number of threads as a hyperparameter to tune for best inference performance: sweep ORT intra-op threads (1, 2, 3, 4) and compare latency/FPS, CPU %, temperature; also inter-op threads and core pinning (`taskset`). First full run used 4 threads.
- FCN8 ONNX has Resize + ArgMax built in (int64 output): may need cutting before Resize/ArgMax for NCNN and Hailo.
- INT8 calibration (#5, #6): use ~200 Cityscapes **train** images, never val. Copy from `leftImg8bit_trainvaltest.zip` on PC.
- [ ] Hailo NPU deployment of FCN8-ResNet18 (same harness)
- [ ] FCN-8s-VGG16 reference (CPU first; Hailo compile once DFC is set up)

## Platform profiling framework (Performance-Profiling-for-Edge-Devices)
- [ ] Integrate the framework (github.com/sameer302/Performance-Profiling-for-Edge-Devices) for proper platform runs: clone as git submodule, per-experiment `experiment_config.json`, `PROTOCOL.md` (§4f), workload CSV with ISO `timestamp`, SIGTERM flush, warm-up, core pinning (monitor core 0, workload cores 1-3), 3 reps with 45-50°C cool-down, run outside Claude Code (§2)
- [ ] Shared workload-side helper in `segmentation/common/` (per-stage timing, ISO timestamps, buffered CSV, SIGTERM flush)
- [ ] Toolkit improvements before large sweeps: end workload phase when workload process exits; non-interactive pre-flight (wait-for-temperature or abort); sweep runner (configs x reps with cool-down)

## Documentation / tooling
- [ ] Write reusable `common/inspect_onnx.py` (validity, IO shapes, params, op types, first/last nodes, GFLOPs) and document how to run it
- [ ] Keep a note of all ONNX inspection scripts/checks used, so they can be rerun manually
- [ ] Maintain a log of all Linux commands used (with short explanations)

## Planning
- [ ] `models.csv`: one row per concrete model with params / GFLOPs measured from model files
- [ ] Next ladder steps: U-Net, DeepLab, STDC, SegFormer
