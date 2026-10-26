# Model weights: FCN8-ResNet18

Weight files are not committed to git (see `segmentation/.gitignore`). This file records where they come from, so the exact model can be restored.

| File | Format | Precision | Source | Size | SHA-256 |
|---|---|---|---|---|---|
| `fcn8_resnet_v1_18.onnx` | ONNX (opset 13, IR 6, exported from PyTorch 1.10) | FP32 | Hailo Model Explorer | 44,811,225 bytes | `ae082c74309cb074ca15c76c4063e8c5a9427dee853ccfadcea6277766e2ff65` |
| `fcn8_resnet_v1_18_mmseg.pth` | PyTorch checkpoint for Hailo's mmsegmentation fork (`{"state_dict", "meta"}`) | FP32 | made from the ONNX file by `../onnx_to_pytorch.py` | 44,905,991 bytes | `75cc947df861d957afff031848baeec4d5639f6c21caad30b8f0c662272e6675` (26 Oct 2026; changes on every rerun, because `meta` holds the creation time) |

The `.pth` is made, not downloaded: run `onnx_to_pytorch.py` (in `venv_torch`) to restore it. It loads into
github.com/hailo-ai/mmsegmentation (branch `Hailo-0.1` or later) with `configs/fcn/fcn8_r18_hailo.py`. The ONNX
export folded the encoder's BatchNorm into the convs, so each encoder BatchNorm in the checkpoint is weight 1,
bias = folded conv bias, mean 0, var 1 − eps: same output as the ONNX file, original BatchNorm statistics lost.
See `../results/onnx_to_pytorch/README.md`.

## Source
- **Hailo Model Explorer (Vision):** https://hailo.ai/products/hailo-software/model-explorer-vision/
- Model: `fcn8_resnet_v1_18`, semantic segmentation, Cityscapes (19 classes). Downloaded as the ONNX (float) file.
- Placed on the Pi on 2026-10-10.
- Licence: check the terms on the Hailo Model Explorer / Hailo Model Zoo before redistributing the weights.

## Restore
1. Open the Model Explorer page above, find `fcn8_resnet_v1_18`, and download the ONNX file.
2. Copy it into this folder (from your PC):
   ```bash
   scp fcn8_resnet_v1_18.onnx sameer@<pi-address>:~/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/segmentation/semantic/01_fcn/fcn8_resnet18/models/
   ```
3. Verify it is the same file used for the results in `../results/`:
   ```bash
   sha256sum fcn8_resnet_v1_18.onnx
   # expected: ae082c74309cb074ca15c76c4063e8c5a9427dee853ccfadcea6277766e2ff65
   ```

## Model facts (from inspecting the ONNX file)
- Input `test_input`: 1×3×1024×1920, float32, NCHW. Expects RGB, normalized with mean [123.675, 116.28, 103.53] and std [58.395, 57.12, 57.375]; normalization is **not** inside the model.
- Output `output`: 1×1×1024×1920, int64 class map (Resize ×8 and ArgMax are inside the model).
- 11.20 M parameters, ≈71 GMACs (≈142 GFLOPs) per image.
