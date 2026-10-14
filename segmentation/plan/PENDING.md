# Pending items

Tick off as they are done. Model under study: FCN8-ResNet18 (ONNX FP32) on Cityscapes val, Pi 5 CPU.

## 1. Next up: memory-management experiments
- [ ] **A. Memory behaviour analysis:** per-operator profiling (ONNX Runtime profiler, time per layer), roofline comparison (compute-bound vs memory-bound layers), measure real RAM bandwidth (STREAM-style benchmark), optionally cache misses with `perf`
- [ ] **B. Fuse the Resize + ArgMax tail:** ArgMax at 1/8 resolution, then nearest upsample of the class map, instead of a 142.5 MiB full-resolution score tensor. Measure latency and mIoU
- [ ] **C. Input resolution:** e.g. 1024×1920 vs 512×1024 (≈4× fewer pixels). Speed vs accuracy trade-off
- [ ] **D. Accuracy options:** evaluate at full 2048 width instead of resizing to 1920; test-time flip augmentation

## 2. Experiment guidelines (ideal conditions)
Follow these for the final, reportable experiments, once there is a concrete finding. Not applied to every exploratory run (too time-consuming).
- [ ] Repeated runs (>= 3) per configuration, report mean ± std across runs
- [ ] Run without Claude Code or other background processes (profiling guidelines §2)
- [ ] Start each run at 45–50 °C, `performance` governor, no throttling (guidelines §2b, §4l)
- [ ] Record the environment: versions (`requirements_cpu.txt`), cooling, ambient temperature, `free -h` (guidelines §4e)

## 3. Future experiments
### Runtime × precision sequence (change one dimension per step)
| # | Runtime | Precision | Compare with | Status |
|---|---|---|---|---|
| 1 | PyTorch | FP32 | #2 (framework effect) | Optional, needs `venv_export` + original checkpoint |
| 2 | ONNX Runtime | FP32 | Published mIoU (pipeline check) | Done (69.87% vs 69.4% ref) |
| 3 | NCNN | FP32 (fp16 disabled) | #2 (runtime effect) | Pending |
| 4 | NCNN | FP16 (A76 native) | #3 (precision effect) | Pending |
| 5 | ONNX Runtime | INT8 static | #2 (quantization effect) | Pending |
| 6 | NCNN | INT8 | #3 (quantization effect) | Pending |

- NCNN enables fp16 by default on Cortex-A76: disable `use_fp16_*` options for the true FP32 run (#3).
- FCN8 ONNX has Resize + ArgMax built in (int64 output): may need cutting before Resize/ArgMax for NCNN and Hailo.
- INT8 calibration (#5, #6): use ~200 Cityscapes **train** images, never val. Copy from `leftImg8bit_trainvaltest.zip` on PC.

### Memory interaction of model customization techniques
FlexNN (MobiCom 2024) argues that model customization (compression, efficient architecture design, NAS) needs intensive development effort and degrades accuracy through reduced model capacity. Apply each technique to the same FCN8-ResNet18 and measure how it changes memory, to see which part of memory (weights or activations) each one addresses.
- [ ] Quantization: INT8 (static) and FP16; overlaps with runtime × precision #4–#6
- [ ] Pruning: unstructured (sparsity only, dense kernels may save nothing) vs structured / channel pruning (smaller tensors)
- [ ] Efficient architecture: same FCN8 decoder with a lighter backbone (e.g. MobileNetV2), links to the backbone sweep
- [ ] NAS: use an existing NAS-derived model (e.g. MobileNetV3, EfficientNet backbone) rather than running a search
- Measure for each: peak RSS, per-layer weight vs activation memory (ORT profiler, as in experiment A), latency, mIoU, and development effort needed (fine-tuning, calibration data)

### Threads and core pinning
First full run used ORT 4 intra-op / 1 inter-op threads, OpenCV default threads, no pinning.

**Thread pools in the pipeline**
| Library | Used for | Control | Impact |
|---|---|---|---|
| ONNX Runtime | Inference (92% of time) | `SessionOptions.intra_op_num_threads`, `inter_op_num_threads` | Dominant |
| OpenCV | Resize / color conversion (pre 3.6%, post 0.2%) | `cv2.setNumThreads(n)` | Small |
| NumPy | Normalization, confusion matrix | Element-wise ops are single-threaded; only BLAS ops use `OPENBLAS_NUM_THREADS` | Negligible |
| OS | Which cores threads run on | `taskset -c <cores>` (process), ORT `session.intra_op_thread_affinities` (ORT threads) | Run-to-run stability |

**Why not every combination**
- Stages run sequentially (load → pre → infer → post), so OpenCV and ORT are never busy at the same time: each can use all 4 cores in its own stage and can be tuned separately.
- Tune in proportion to time spent: ORT threads decide almost everything; perfect OpenCV tuning can save at most part of 66 ms (< 4%).
- Only 4 cores: the full ORT × OpenCV grid is 16 configs (~1 h on a 100-image subset), affordable but usually unnecessary.

**Where libraries do interfere**
- ORT spin-waiting: after each inference its threads keep spinning briefly, which can slow the next image's OpenCV pre-processing. Test `so.add_session_config_entry("session.intra_op.allow_spinning", "0")`.
- Parallel / pipelined stages (real-time camera pipeline: capture + pre of frame N+1 during inference of frame N): pools compete, so split cores between stages (e.g. inference 3 cores, capture/pre 1 core), pin each, keep total threads ≤ 4.
- Other processes (monitor thread, Claude Code, desktop) share cores: pinning keeps them off inference cores.

**Procedure**
- [ ] 1. ORT sweep: intra-op threads 1, 2, 3, 4 (OpenCV at default), each with spinning on and off; compare latency/FPS, CPU %, temperature
- [ ] 2. OpenCV sweep: set `cv2.setNumThreads` explicitly (currently left at OpenCV default) to 1, 2, 4 with the best ORT setting
- [ ] 3. Interaction check: small grid around the best pair; if no change, stages are confirmed independent
- [ ] 4. Core pinning: unpinned vs pinned (`taskset`, e.g. workload on cores 1–3 or 0–3)
- [ ] 5. If a real-time pipelined version is built later: redo as a core-allocation problem across stages
- Expectation to verify: ORT 4 threads at or near best; OpenCV setting changes end-to-end time by < 2%

### Other platforms and models
- [ ] Hailo NPU deployment of FCN8-ResNet18 (same evaluation harness)
- [ ] FCN-8s-VGG16 reference: CPU first; Hailo after compiling the HEF manually with the DFC (x86)
- [ ] Next ladder steps: U-Net → DeepLab → STDC → SegFormer
- [ ] `models.csv`: one row per model with params / GFLOPs measured from the model files

### Possible medical-imaging track (for a biomedical venue)
Develop the memory method generally (Cityscapes, comparable with systems work like FlexNN), then add a medical evaluation; choose the venue by where the stronger result is.
- [ ] Datasets: one 2D (e.g. Kvasir-SEG polyps, freely available; or ISIC skin lesions) and later one 3D (e.g. BraTS brain MRI, needs registration)
- [ ] Models: U-Net family (U-Net, nnU-Net, lightweight e.g. UNeXt; later TransUNet, MedSAM/MobileSAM). U-Net is also the next ladder step, so it bridges both tracks
- [ ] Metrics: Dice, HD95, sensitivity, plus latency, memory, energy; show any accuracy drop from quantization/memory tricks is clinically negligible
- [ ] 2D vs 3D: 3D volumes have far larger activations, so activation-heavy memory management matters most there
- Rationale: low cost / low-resource point-of-care, privacy (images stay on device), real-time (endoscopy, ultrasound)
- Venues to check (verify current calls for papers): IEEE EMBC, ISBI, MICCAI workshops, IEEE JBHI, Computers in Biology and Medicine; systems alternative: MobiSys/SenSys, IEEE ESL, ACM TECS, JSA, DATE/DAC

## 4. Platform profiling framework integration
- [ ] Integrate the framework (github.com/sameer302/Performance-Profiling-for-Edge-Devices): clone as git submodule, per-experiment `experiment_config.json`, `PROTOCOL.md` (§4f), workload CSV with ISO `timestamp`, SIGTERM flush, warm-up, core pinning, 3 reps with cool-down, run outside Claude Code (§2)
- [ ] Shared workload-side helper in `segmentation/common/` (per-stage timing, ISO timestamps, buffered CSV, SIGTERM flush)
- [ ] Toolkit: log memory in MiB (2^20 bytes, as `free -m`) instead of MB (10^6) in `platform_monitor.py`, to match `run_eval.py` (switched to MiB on 2026-10-13)
- [ ] Toolkit improvements before large sweeps: end workload phase when the workload process exits; non-interactive pre-flight (wait-for-temperature or abort); sweep runner (configs × reps with cool-down)

## 5. Model format inspection scripts
- [ ] Reusable `common/inspect_onnx.py` (validity, IO shapes, params, op types, first/last nodes, GFLOPs) with run instructions
- [ ] Keep a note of every inspection check used, so it can be rerun manually
- [ ] Equivalent inspection for other formats as they are added (NCNN `.param/.bin`, Hailo `.hef`)
