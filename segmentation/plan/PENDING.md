# Pending items

Tick off as they are done. Model under study: FCN8-ResNet18 (ONNX FP32) on Cityscapes val, Pi 5 CPU.

## 1. Next up: memory-management experiments
- [ ] **Next session, in this order (agreed 2026-10-22):**
  1. [x] **Arena check (done 2026-10-26, `arena_log_check.py`, results/arena_log_check/run_20261026_122559):** run 1 takes 3 regions (128 for the 120 MiB stem output, 128 for the 30 MiB max-pool output, 256 for the 142.5 MiB Resize output) = 513 MiB reserved, RSS +323; run 2 takes no new region in either setting, RSS rises only by newly written pages (+7.5 pattern on, +60 off here; +56 / +105 in the settings experiment). Explanation slide added before the run-by-run slide. Original plan: run arena on (pattern on and off) with verbose ORT logging, log each arena extension (requested vs region size) in runs 1 and 2, and compare with the RSS readings. Expected: arena grows in doubling regions, merges neighbouring free pieces, best fit; RSS counts only touched pages. If it matches, write the explanation for slide 11 (run 2 +56 MiB on/on, +105 MiB on/off); if not, leave it (many factors affect these readings)
  2. **Temporary buffers + allocator slack per fused layer** (step 0b below): arena and pattern off, **malloc threshold fixed (`MALLOC_MMAP_THRESHOLD_=131072`)**: the malloc check (2026-10-26, `results/ort_mem_settings/malloc_threshold_check/`) showed this cuts the slack to 1–3 MiB in every run (default 20–54 MiB), so measured − activations ≈ pure temporary buffers; leftover = measured − weights − all live activations − fixed overhead; then one combined per-layer chart: weights + activations + buffers/slack
  3. Continue reading FlexNN, then try a similar concept for activations
  4. **Write a manuscript** of the work so far for the PI to review
- [x] **A. Memory behaviour analysis:** per-operator profiling (ONNX Runtime profiler, time per layer), roofline comparison (compute-bound vs memory-bound layers), measure real RAM bandwidth (STREAM-style benchmark), optionally cache misses with `perf`
  - Done 2026-10-15 (exploratory): `profile_layers.py`, `layer_stats.py`, `analyze_profile.py`, STREAM in `common/bench/stream/`; results in `results/exp_a_profile/run_20261015_115052/`. Decoder 7.5% of time vs 0.2% of MACs; 21 compute-bound / 9 memory-bound layers; RAM 8.2 GB/s (4 threads); activation peak 157.5 MiB vs 42.7 MiB weights; gap to peak: 37% conv kernels, 9% memory-related
  - [ ] Still open: clean-protocol repeat before reporting; `perf` cache misses (optional); encoder/decoder split cross-check (optional); why ArgMax is slow
- [ ] **B. Fuse the Resize + ArgMax tail:** ArgMax at 1/8 resolution, then nearest upsample of the class map, instead of a 142.5 MiB full-resolution score tensor. Measure latency and mIoU
  - Exp A (2026-10-15): Resize 27.8 ms + ArgMax 83.7 ms = 111.5 ms (6.5% of time); ArgMax reaches only 2.0 GB/s vs 8.2 GB/s RAM (STREAM, 4 threads), so its kernel is the bottleneck (~64 ms lost). Also check why: single-threaded? class-plane access order? Compare with a NumPy / custom argmax on the same tensor
- [ ] **C. Input resolution:** e.g. 1024×1920 vs 512×1024 (≈4× fewer pixels). Speed vs accuracy trade-off
- [ ] **D. Accuracy options:** evaluate at full 2048 width instead of resizing to 1920; test-time flip augmentation

## 2. Experiment guidelines (ideal conditions)
Follow these for the final, reportable experiments, once there is a concrete finding. Not applied to every exploratory run (too time-consuming).

**Exploratory runs** (current setup: Claude Code open, no pinning): use the median not the mean, 3 repeats when a run is short, log CPU load / temperature / clock before each run, run compared variants back to back in the same session. Switch to the clean protocol when a number becomes a final result (paper, supervisor) or the effect is small (< ~5%, comparable to the noise). Static results (FLOPs, tensor sizes) and time shares are barely affected by background load; absolute times, sub-millisecond layers and RAM bandwidth are.
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

### Reducing activation memory
Exp A (2026-10-15): activations, not weights, set the memory peak (157.5 MiB activations vs 42.7 MiB weights). The two largest peaks are at opposite ends: decoder tail 157.5 MiB at ArgMax (Resize output 142.5 + int64 label map 15) and stem 150 MiB; all other blocks ≤ 90 MiB. FlexNN-style weight slicing does not touch these, so a segmentation-specific activation plan is a possible research contribution (literature check on tiled / patch-based inference for dense prediction needed before claiming novelty).
- [x] **Effect of ORT memory settings** (done 2026-10-21, exploratory; whole model only, per-layer moved to 0b): `ort_mem_settings.py`, `plot_mem_settings.py`, results in `results/ort_mem_settings/run_20261021_151457/`. 4 combinations of `enable_cpu_mem_arena` × `enable_mem_pattern`, fresh process each, 3 repeats × 10 runs, separate passes for RAM over time (sampler on, it costs ~5% latency) and latency (sampler off). Latency all within 2% (1,680–1,713 ms, no real difference, ~1% above baseline). Peak / between-run RAM: on/on 602 / 601, on/off 654 / 653 (unexpected), off/on 397 / 248, off/off ~400 / 261 MiB. Arena off + pattern on: −205 MiB peak (−34%) at the same speed
  - [ ] Clean-protocol repeat before reporting; why on/off holds more than on/on: answered 2026-10-26 (no new region in run 2; newly written pages, arena_log_check.py)
- [x] 0. **Runtime overhead** (done 2026-10-21, from the stage readings of the settings experiment; identical in all 4 settings up to the start of run 1, since the settings only change activation allocation). Fixed overhead before run 1: 157 MiB (108 without test-only parts) = Python 12.8 + NumPy/OpenCV 41.6 + ORT library 19.3 + ORT session structures ~4.7 + pre-processing leftovers 29.3 + test-only 48.9; plus weights 42.7 and input 22.5 listed separately (total 222). Inference adds a setting-dependent overhead: ~222 MiB arena overhead in the default, ~26 MiB kept between runs with the arena off. Default setting, 602 MiB in total: Python 12.8, NumPy + OpenCV 41.6, ONNX Runtime library 19.3, test-only helpers (onnx, psutil) 11.4, test-only 5 raw images 37.5, session (weights 42.7 + ORT structures ~4.7) 47.4, input + pre-processing leftovers 51.8 → 222 before inference; then activations (157.5 computed) + arena overhead (~222) = 380. Library code (file-backed) ~57 MiB of the total. Weights are not kept twice. Arena off removes the ~222 MiB allocator overhead (peak 397). Run-1 peak with arena off is 158 MiB above the pre-run level ≈ 157.5 MiB computed, so buffers are small at the peak moment
- [ ] 0b. **Temporary buffers per fused layer:** memory per layer = weights + activations (computed, `layer_stats.py`) + kernel temporary buffers (im2col, Winograd; not computed) + runtime overhead (step 0). FlexNN's per-layer Fig. 1 is measured in NCNN and includes buffers, so a fair comparison needs them. ORT (NEXT): arena and pattern both off, RSS sampled ~1 ms during inference, samples matched to fused layers by profiler timestamps, also per-layer latency; buffers = measured − computed live activations. (Cutting the model after each layer was dropped: a cut model's peak is the max over all earlier layers, so smaller later layers are hidden.) NCNN (after the port): custom allocator logging every allocation per layer. Check the stem 7×7 conv first: a full im2col would be ≈ 276 MiB
- [ ] 1. **uint8 label map:** output uint8 instead of int64 (19 classes fit), 15 → 1.9 MiB. No accuracy cost
- [ ] 2. **ArgMax at 1/8 resolution + nearest upsample** (= experiment B): the 142.5 MiB Resize output is never created; peak moves to the stem (~150 MiB). Measure mIoU loss at object edges
- [ ] 3. **Fused, tiled Resize + ArgMax:** upsample and argmax one strip of rows at a time; same saving as 2 with exact results. Needs a custom kernel (NumPy / C) outside ORT
- [ ] 4. **Spatial tiling of the early layers:** run stem + layer1 on image tiles with an overlap margin (receptive field) so outputs match; activations shrink by about the number of tiles, at the cost of recomputed margins. Patch-based inference idea from MCUNetV2 (Lin et al., NeurIPS 2021, microcontrollers). Estimate with 1–3 and 4 tiles: ~60 MiB peak, set by layer2
- [ ] 5. **FP16 / INT8 activations:** 2× / 4× smaller; shares runs with runtime × precision #4–#6
- Estimates above are calculated from `layer_stats.py` tensor sizes, not measured. Combining 2/3 + 4 + 5 could bring the activation peak from 157.5 MiB to roughly 15–30 MiB
- Measure for each: peak RSS, computed activation peak, latency, mIoU

### Comparing with FlexNN (when a method exists)
- FlexNN (X. Li et al., ACM MobiCom 2024) is built on NCNN; on the Raspberry Pi it computes on 3 cores and reserves the 4th for its weight-loading thread; all its baselines also compute on 3 cores (§5.1). Memory measured with `pmap`
- [ ] For a fair comparison use the same setup: 3 compute threads, 4th core free (or used by a method's helper thread), same device, same runtime (NCNN) for both methods
- [ ] Check whether the FlexNN code is public; otherwise compare against reported numbers only (weaker)

### Cross-runtime memory comparison
FlexNN (MobiCom 2024) measures only NCNN variants; MNN/NCNN memory behaviour is stated in its introduction but not measured, TFLite is only called slower, ONNX Runtime and OpenVINO are not mentioned. Run the same FCN8-ResNet18 (FP32) on several runtimes and compare how each one manages memory.
- [ ] Runtimes: ONNX Runtime (done, baseline), NCNN, TFLite / LiteRT (XNNPACK); OpenVINO only if its ARM CPU plugin works on the Pi 5
- [ ] Compare: weight residency (all weights loaded up front?), activation memory pool / arena planning, kernel choice and its memory cost (Winograd, im2col + GEMM, direct)
- [ ] Runtime memory options to test: ORT `enable_cpu_mem_arena`, `enable_mem_pattern`; NCNN `use_winograd_convolution`, `use_sgemm_convolution`, `lightmode`
- [ ] Measure: peak RSS, per-layer memory where the runtime exposes it, latency, mIoU (must match across runtimes)
- Shares runs with the runtime × precision sequence (#2, #3), so only extra memory logging is needed

### Comparing runtimes layer by layer
**Difficulty:** each runtime merges (fuses) layers differently, so per-layer results do not line up across runtimes. ONNX Runtime fuses Conv+ReLU, NCNN may fuse Conv+BN+ReLU, Hailo merges whole groups of layers into hardware contexts. Layer names and counts differ, so "layer 23" in one runtime is not "layer 23" in another.

**Per-layer profiling tools per runtime**
| Runtime | Per-layer time | Memory |
|---|---|---|
| ONNX Runtime | `enable_profiling` (JSON trace per operator) | Tensor sizes from model; process RSS |
| NCNN | Build with `-DNCNN_BENCHMARK=ON`: prints time, type and shapes per layer | Custom allocator can log every allocation |
| TFLite / LiteRT | `benchmark_model --enable_op_profiling=true` | `benchmark_model` peak memory report |
| OpenVINO | `benchmark_app -pc` (performance counters) | Total only |
| PyTorch (optional) | `torch.profiler` | `profile_memory=True` |
| Hailo NPU | DFC profiler: per-layer **estimate** from compilation; on device `hailortcli` gives whole-model latency / FPS / power only | Weights on chip; host RAM holds input / output only |

**Plan to solve it**
- [ ] Compare runtimes at **block level** (stem, layer1–4, score / upsample, Resize, ArgMax), which exists in every runtime; compare individual layers only within one runtime
- [ ] Map each runtime's (fused) layers to blocks, using the original ONNX names in `architecture/layer_table.csv` as the reference; record which original layers each fused layer contains
- [ ] Common per-layer CSV format for all runtimes: `layer, op_type, block, time_ms`; one small reader per runtime converts its profiler output to it, so the analysis and plots (`analyze_profile.py`) are reused unchanged
- [ ] Runtime-independent quantities computed once from the model file (`layer_stats.py`): FLOPs, weight / activation sizes, peak memory estimate; only bytes per value change with precision (FP32 4, FP16 2, INT8 1)
- [ ] Fallback when a runtime has no per-layer profiler (e.g. Hailo on device): cut the model after each block, time each prefix, block time = difference between consecutive prefixes
- [ ] Runtime-independent cross-checks: RSS sampled over time, `perf` cache misses
- [ ] Move the shared parts (block mapping, common CSV format, `layer_stats.py`, `analyze_profile.py`) to `segmentation/common/` once a second runtime is added

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
- [ ] 6. **Per-layer thread count** (from the exp A STREAM finding below):
  - [ ] a. Measure: run `profile_layers.py` with intra-op threads 1, 2, 3, 4 (currently fixed at 4; add a `--threads` option) and compare each layer's median time across settings; expect MaxPool / Resize / ArgMax to prefer fewer threads, 3x3 convs to need 4
  - [ ] b. Apply only if a. shows a gain: ORT has no per-layer thread setting, so split the model into sub-models with their own sessions / thread counts, or run the tail (Resize + ArgMax) outside ORT with its own thread count; check that the split adds no copy overhead larger than the gain
- Expectation to verify: ORT 4 threads at or near best; OpenCV setting changes end-to-end time by < 2%
- Exp A finding (STREAM, 2026-10-15): RAM bandwidth falls with more threads (1 thread 11.9, 2 threads 10.0, 4 threads 8.2 GB/s). Memory-bound layers (MaxPool, Resize, ArgMax) may run as fast or faster on fewer threads, while 3x3 convs need all 4. Check per-layer time vs thread count in the ORT sweep; a per-layer thread count could help

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
