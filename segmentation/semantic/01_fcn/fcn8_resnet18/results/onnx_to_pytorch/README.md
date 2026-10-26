# ONNX → PyTorch checkpoint (26 Oct 2026)

Why: a batchmate will apply model compression (before and after training) in PyTorch. No PyTorch checkpoint
of `fcn8_resnet_v1_18` is published; Hailo publishes only the training code, its fork of mmsegmentation
(github.com/hailo-ai/mmsegmentation, branch `Hailo-0.1` or later, config `configs/fcn/fcn8_r18_hailo.py`,
head `mmseg/models/decode_heads/fcn_generalized_head.py`). Our ONNX graph matches that config layer for layer,
and its decoder weights still carry the mmseg names, so `../../onnx_to_pytorch.py` copies the ONNX weights
into a checkpoint for that fork (`../../models/fcn8_resnet_v1_18_mmseg.pth`, not committed, see
`../../models/README.md`).

Environment: `venv_torch` at the repo root (separate from `venv_cpu`, so the ONNX Runtime measurements are
unchanged): torch 2.14.1+cpu, torchvision 0.29.1+cpu (CPU wheels from download.pytorch.org/whl/cpu) and
`requirements_cpu.txt`. Like every venv it holds its own `.gitignore`, so it stays out of git.

```bash
python3 -m venv venv_torch
venv_torch/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv_torch/bin/pip install -r requirements_cpu.txt
cd segmentation/semantic/01_fcn/fcn8_resnet18 && ../../../../venv_torch/bin/python onnx_to_pytorch.py
```

## Result (`run_20261026_183919/verify.json`)

ONNX Runtime 1.30 vs PyTorch 2.14 (CPU, FP32, 4 threads), one random input and the first 5 Cityscapes val images:

| Input | Decoder output, max abs diff | Logits after Resize, max abs diff | Class map |
|---|---|---|---|
| random | 1.1e-5 (max logit 12.2) | 6.2e-5 | 2 of 1,966,080 px differ |
| 5 val images | 4.7e-5 – 8.4e-5 (max logit 17–21) | 1.2e-4 – 1.5e-4 | 0–2 px differ |

The differences are floating-point rounding (different kernels, ~1e-5 of the logit size); the few differing
pixels are places where two classes have almost equal logits. The checkpoint gives the same model as the ONNX file.

## Limits
- Encoder BatchNorm was folded into the convs by the ONNX export. The checkpoint carries it as BatchNorm with
  weight 1, bias = conv bias, mean 0, var 1 − eps. Inference is identical; in training mode BatchNorm uses batch
  statistics, so fine-tuning starts slightly off the original model and recovers. Compression after training
  (no retraining) is not affected.
- Checked in a stand-in PyTorch model with the same module names (`FCN8ResNet18` in the script), not in the
  mmseg fork itself (mmcv-full 1.x is not installed on the Pi). Loading into the fork is the first thing to
  check on the batchmate's machine (`load_checkpoint(..., strict=True)` should report no missing keys).
- The config uses `SyncBN`; on one GPU change it to `BN` (same state_dict keys).

## For the batchmate
Full step-by-step guide: "FCN8-ResNet18 compression guide" (claude.ai/artifact/XcndVimzYEMJ6t9dnrPC61, private
until shared). In short:
1. Build Hailo's training Docker image (`hailo_model_zoo/training/fcn`; it uses branch `Hailo-1.0` of the fork,
   mmcv 2.0.0, mmengine 0.8.5) and download Cityscapes yourself (cityscapes-dataset.com; the licence does not allow
   us to share it), mounted at `/data/cityscapes`.
2. Export the unmodified checkpoint with Hailo's `tools/pytorch2onnx.py` (`--shape 1024 1920`; it loads
   `ckpt['state_dict']` strictly) and check it gives 69.87% mIoU with our `run_eval.py`.
3. Compress (with `load_from = 'fcn8_resnet_v1_18_mmseg.pth'` when fine-tuning), export each model the same way
   and send the ONNX files with their `run_eval.py` mIoU and method.

Files to send: the `.pth`, the original ONNX, `run_eval.py`, `requirements_cpu.txt`.
