#!/usr/bin/env python3
"""Turn the Hailo ONNX model back into a PyTorch checkpoint for Hailo's mmsegmentation fork.

There is no public PyTorch checkpoint of fcn8_resnet_v1_18. Hailo trains it with its fork of mmsegmentation
(github.com/hailo-ai/mmsegmentation, branch Hailo-0.1 or later, config configs/fcn/fcn8_r18_hailo.py):
  backbone     mmseg ResNet-18 (pytorch style, strides 1,2,2,2, no dilation), BatchNorm after every conv
  decode_head  FCNGenHead, in_index [1,2,3] (layer2/3/4 outputs, 128/256/512 ch), num_convs 0, 19 classes:
               conv_seg (1x1 on layer4) -> upsample_head1 (ConvTranspose 4x4 s2 + BN) + cls_head1 (1x1 on layer3)
               -> upsample_head0 (ConvTranspose + BN) + cls_head0 (1x1 on layer2)
  inference    bilinear resize to the input size (align_corners=True), no softmax; Hailo's export adds ArgMax
The ONNX graph has exactly these layers, and the head weights still carry their mmseg names
("model.decode_head.upsample_head1.0.weight", ...). In the backbone the export folded each BatchNorm into
its conv, so the backbone weights are anonymous and every conv has a bias. mmseg's backbone convs have no
bias, so the folded bias is put into the BatchNorm after the conv, which then passes the conv output
through and adds the bias: weight 1, bias b, running_mean 0, running_var 1 - eps (sqrt(var + eps) = 1).
The output is exactly the ONNX model; the original BatchNorm statistics cannot be recovered.

The PyTorch model below is only a stand-in for the mmseg one (mmseg is not installed on the Pi): same
modules, same state_dict keys, so the saved checkpoint loads into the fork with fcn8_r18_hailo.py.
Backbone convs are named by their order in the graph (conv1, then per block conv1, conv2 and, in the first
block of layer2-4, downsample.0), and every shape is checked against the PyTorch model.

Checks, ONNX Runtime vs PyTorch (CPU, FP32, eval mode), on one random input and N Cityscapes val images:
  head logits before the resize (1x19x128x240), logits after the resize (1x19x1024x1920): max abs difference
  class map (ArgMax): share of pixels with the same class

Outputs:
  models/fcn8_resnet_v1_18_mmseg.pth   mmseg checkpoint {"state_dict", "meta"}; not committed (licence)
  results/onnx_to_pytorch/run_<timestamp>/verify.json

Needs torch + torchvision, onnx, onnxruntime, opencv (venv_torch, see results/onnx_to_pytorch/README.md).

Usage:
  python onnx_to_pytorch.py                    # convert + check on 5 val images
  python onnx_to_pytorch.py --num-images 20
"""
import argparse
import glob
import json
import os
import platform
from datetime import datetime

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from onnx import numpy_helper

from run_eval import preprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
NUM_CLASSES = 19
BN_EPS = 1e-5
CITYSCAPES_CLASSES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                      "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                      "motorcycle", "bicycle")


class FCNGenHead(nn.Module):
    """FCNGenHead of Hailo's mmseg fork with num_convs=0, concat_input=False, in_index [1,2,3]."""

    def __init__(self, in_channels=(128, 256, 512)):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(in_channels[-1], NUM_CLASSES, 1)
        for i, c in enumerate(in_channels[:-1]):
            self.add_module(f"upsample_head{i}", nn.Sequential(
                nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, 4, 2, padding=1), nn.BatchNorm2d(NUM_CLASSES)))
            self.add_module(f"cls_head{i}", nn.Conv2d(c, NUM_CLASSES, 1))

    def forward(self, feats):
        out = self.conv_seg(self.dropout(feats[-1]))
        for i in reversed(range(len(feats) - 1)):      # layer3 first, then layer2
            out = getattr(self, f"upsample_head{i}")(out) + getattr(self, f"cls_head{i}")(feats[i])
        return out


class FCN8ResNet18(nn.Module):
    """Same modules and state_dict keys as mmseg EncoderDecoder(ResNet-18, FCNGenHead)."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)   # same key names as mmseg ResNet
        del self.backbone.fc, self.backbone.avgpool
        self.decode_head = FCNGenHead()

    def forward(self, x, return_logits=False):
        b = self.backbone
        y = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        feats = []
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            y = layer(y)
            feats.append(y)
        head = self.decode_head(feats[1:])
        logits = F.interpolate(head, size=x.shape[2:], mode="bilinear", align_corners=True)
        if return_logits:
            return head, logits
        return logits.argmax(1, keepdim=True)


def backbone_conv_names():
    names = ["conv1"]
    for layer in range(1, 5):
        for block in range(2):
            p = f"layer{layer}.{block}"
            names += [f"{p}.conv1", f"{p}.conv2"]
            if block == 0 and layer > 1:
                names.append(f"{p}.downsample.0")
    return names


def bn_name(conv):
    if conv.endswith("downsample.0"):
        return conv[:-1] + "1"
    return conv[:-len("conv1")] + "bn" + conv[-1]


def onnx_to_state_dict(model_path, net):
    graph = onnx.load(model_path).graph
    inits = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in graph.initializer}
    target = net.state_dict()
    sd = {}

    convs = [n for n in graph.node if n.op_type == "Conv" and not n.input[1].startswith("model.decode_head")]
    names = backbone_conv_names()
    assert len(convs) == len(names), f"{len(convs)} backbone convs in ONNX, expected {len(names)}"
    for node, name in zip(convs, names):
        w, bias = inits[node.input[1]], inits[node.input[2]]
        key, bn = f"backbone.{name}", f"backbone.{bn_name(name)}"
        assert w.shape == target[f"{key}.weight"].shape, (name, w.shape, target[f"{key}.weight"].shape)
        sd[f"{key}.weight"] = w
        sd[f"{bn}.weight"] = torch.ones_like(bias)
        sd[f"{bn}.bias"] = bias
        sd[f"{bn}.running_mean"] = torch.zeros_like(bias)
        sd[f"{bn}.running_var"] = torch.full_like(bias, 1.0 - BN_EPS)
        sd[f"{bn}.num_batches_tracked"] = torch.tensor(0)

    for name, t in inits.items():
        if name.startswith("model.decode_head."):
            key = name[len("model."):]
            assert t.shape == target[key].shape, (key, t.shape, target[key].shape)
            sd[key] = t
    for key in target:                                  # head BN counters are not in the ONNX file
        if key.endswith("num_batches_tracked") and key not in sd:
            sd[key] = torch.tensor(0)

    missing, extra = set(target) - set(sd), set(sd) - set(target)
    assert not missing and not extra, f"missing {sorted(missing)}, extra {sorted(extra)}"
    eps = {m.eps for m in net.modules() if isinstance(m, nn.BatchNorm2d)}
    assert eps == {BN_EPS}, eps
    return sd


def ort_session(model_path, threads):
    """ONNX model with two extra outputs: head logits (input of the Resize) and logits after the Resize."""
    m = onnx.load(model_path)
    resize = next(n for n in m.graph.node if n.op_type == "Resize")
    extra = [resize.input[0], resize.output[0]]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(n) for n in extra)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess, extra


def compare(sess, extra, net, x):
    names = [sess.get_outputs()[0].name] + extra
    ort_map, ort_head, ort_logits = sess.run(names, {sess.get_inputs()[0].name: x})
    with torch.no_grad():
        head, logits = net(torch.from_numpy(x), return_logits=True)
    pt_map = logits.argmax(1, keepdim=True).numpy()
    return {
        "head_max_abs_diff": float(np.abs(head.numpy() - ort_head).max()),
        "head_max_abs_value": float(np.abs(ort_head).max()),
        "logits_max_abs_diff": float(np.abs(logits.numpy() - ort_logits).max()),
        "class_map_agreement": float((pt_map == ort_map).mean()),
        "pixels_different": int((pt_map != ort_map).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    ap.add_argument("--out-ckpt", default=os.path.join(HERE, "models/fcn8_resnet_v1_18_mmseg.pth"))
    ap.add_argument("--data-root", default=os.path.join(REPO, "segmentation/datasets/cityscapes"))
    ap.add_argument("--out-dir", default=None, help="default: results/onnx_to_pytorch/run_<timestamp>")
    ap.add_argument("--num-images", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(HERE, "results/onnx_to_pytorch",
                                           datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(args.threads)

    net = FCN8ResNet18().eval()
    sd = onnx_to_state_dict(args.model, net)
    net.load_state_dict(sd, strict=True)
    meta = {"CLASSES": CITYSCAPES_CLASSES, "source": os.path.basename(args.model),
            "config": "hailo-ai/mmsegmentation configs/fcn/fcn8_r18_hailo.py",
            "note": "backbone BatchNorm folded in the ONNX export; carried as BN(weight=1, bias=conv bias, "
                    "mean=0, var=1-eps)", "created": datetime.now().isoformat()}
    torch.save({"state_dict": sd, "meta": meta}, args.out_ckpt)
    print(f"checkpoint: {args.out_ckpt} ({os.path.getsize(args.out_ckpt) / 2**20:.1f} MiB, {len(sd)} tensors)")

    sess, extra = ort_session(args.model, args.threads)
    _, _, in_h, in_w = sess.get_inputs()[0].shape
    cases = [("random", np.random.default_rng(0).standard_normal((1, 3, in_h, in_w), dtype=np.float32))]
    images = sorted(glob.glob(os.path.join(args.data_root, "leftImg8bit/val/*/*_leftImg8bit.png")))
    for path in images[:args.num_images]:
        cases.append((os.path.basename(path), preprocess(cv2.imread(path), in_w, in_h)))

    results = []
    for name, x in cases:
        r = {"input": name, **compare(sess, extra, net, x)}
        results.append(r)
        print(f"{name:45s} head diff {r['head_max_abs_diff']:.2e} (max |logit| {r['head_max_abs_value']:.1f})  "
              f"logits diff {r['logits_max_abs_diff']:.2e}  class map {100 * r['class_map_agreement']:.4f}% same "
              f"({r['pixels_different']} px differ)")

    summary = {"model": args.model, "checkpoint": args.out_ckpt, "results": results,
               "versions": {"python": platform.python_version(), "torch": torch.__version__,
                            "torchvision": torchvision.__version__, "onnx": onnx.__version__,
                            "onnxruntime": ort.__version__, "numpy": np.__version__},
               "threads": args.threads, "timestamp": datetime.now().isoformat()}
    with open(os.path.join(out_dir, "verify.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
