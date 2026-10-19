#!/usr/bin/env python3
"""Experiment A, script 2: computation and memory of every layer, from the model files (no inference).

Analyses the optimised graph saved by profile_layers.py (the layers ONNX Runtime actually runs), so
layer names match the profiler. Tensor shapes come from ONNX shape inference on the original model
(every tensor name in the optimised graph also exists there). Each original layer that ONNX Runtime
fused away (ReLU, Add) is assigned to the fused layer that absorbed it, and each layer gets the
block of its original layer from architecture/layer_table.csv.

Counting conventions (FP32 = 4 bytes, int64 = 8 bytes):
  Conv / FusedConv   MACs = C_out * H_out * W_out * (C_in / groups) * k_h * k_w;  FLOPs = 2 * MACs
                     + 1 FLOP per output element each for bias, fused Add and fused ReLU
  ConvTranspose      MACs = C_in * H_in * W_in * (C_out / groups) * k_h * k_w;    FLOPs = 2 * MACs + bias
  BatchNormalization 2 FLOPs per element (scale and shift)
  MaxPool            k_h * k_w - 1 comparisons per output element
  Resize (bilinear)  ~8 operations per output element (approximation)
  ArgMax             C - 1 comparisons per output element
  bytes moved        activations read + weights read + activations written, each once (lower bound:
                     a layer whose data does not fit in cache moves at least this much through RAM)
  live memory        activation tensors alive while a layer runs, in execution order: a tensor lives from
                     the layer that writes it until its last reader. Weights are counted separately.
                     A lower bound: it ignores buffer reuse limits, alignment and fragmentation.

Outputs (in the run folder):
  layer_stats.csv    per layer: block, fused original layers, shapes, MACs, FLOPs, weight / activation
                     sizes, bytes moved, arithmetic intensity, live activation memory
  block_stats.csv    per block: MACs, FLOPs, shares, weights, bytes moved, largest live memory
  layer_stats.json   totals, activation peak and the tensors alive at the peak, checks

Usage:
  python layer_stats.py                           # latest results/exp_a_profile/run_*
  python layer_stats.py --run-dir results/exp_a_profile/run_20261015_115052
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
MIB = 2 ** 20
DTYPE_BYTES = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.INT64: 8,
               onnx.TensorProto.INT32: 4, onnx.TensorProto.INT8: 1, onnx.TensorProto.UINT8: 1}
# Coarse blocks used for all comparisons (layer_table.csv blocks -> group).
GROUP = {"stem": "stem", "layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4",
         "score_32": "fcn_head", "up_32to16": "fcn_head", "score_16": "fcn_head", "add_16": "fcn_head",
         "up_16to8": "fcn_head", "score_8": "fcn_head", "add_8": "fcn_head",
         "resize_x8": "resize", "argmax": "argmax"}
GROUP_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "fcn_head", "resize", "argmax"]


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def tensor_info(model):
    """name -> (shape tuple, bytes per element) for every activation in the shape-inferred model."""
    info = {}
    for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        t = v.type.tensor_type
        info[v.name] = (tuple(d.dim_value for d in t.shape.dim), DTYPE_BYTES.get(t.elem_type, 4))
    return info


def nbytes(info, name):
    shape, b = info[name]
    return int(np.prod(shape)) * b


def count_flops(node, shapes, weights):
    """(MACs, FLOPs) of one node, using the conventions in the module docstring."""
    op = node.op_type
    out_shape = shapes[node.output[0]][0]
    out_elems = int(np.prod(out_shape))
    if op in ("Conv", "FusedConv"):
        w = weights[node.input[1]]
        c_out, c_in_g, kh, kw = w.shape
        macs = out_elems * c_in_g * kh * kw
        flops = 2 * macs
        if len(node.input) > 2 and node.input[2]:
            flops += out_elems                       # bias
        if op == "FusedConv":
            if len(node.input) > 3 and node.input[3]:
                flops += out_elems                   # fused Add (skip connection)
            if attr(node, "activation"):
                flops += out_elems                   # fused ReLU
        return macs, flops
    if op == "ConvTranspose":
        w = weights[node.input[1]]                   # (C_in, C_out / groups, kh, kw)
        c_in, c_out_g, kh, kw = w.shape
        in_shape = shapes[node.input[0]][0]
        macs = c_in * in_shape[2] * in_shape[3] * c_out_g * kh * kw
        return macs, 2 * macs + (out_elems if len(node.input) > 2 and node.input[2] else 0)
    if op == "BatchNormalization":
        return 0, 2 * out_elems
    if op == "MaxPool":
        kh, kw = attr(node, "kernel_shape")
        return 0, out_elems * (kh * kw - 1)
    if op == "Resize":
        return 0, 8 * out_elems
    if op == "ArgMax":
        c = shapes[node.input[0]][0][attr(node, "axis", 0)]
        return 0, out_elems * (c - 1)
    if op in ("Add", "Relu"):
        return 0, out_elems
    raise ValueError(f"no FLOP rule for {op} ({node.name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="profile_layers.py output folder (default: latest)")
    ap.add_argument("--original", default=os.path.join(HERE, "models/fcn8_resnet_v1_18.onnx"))
    ap.add_argument("--layer-table", default=os.path.join(HERE, "architecture/layer_table.csv"))
    args = ap.parse_args()

    run_dir = args.run_dir or sorted(glob.glob(os.path.join(HERE, "results/exp_a_profile/run_*")))[-1]
    original = onnx.load(args.original)
    shapes = tensor_info(shape_inference.infer_shapes(original))
    opt = onnx.load(os.path.join(run_dir, "optimized_model.onnx"))
    weights = {i.name: numpy_helper.to_array(i) for i in opt.graph.initializer}

    # Block of every original layer (layer_table.csv rows are in original node order).
    with open(args.layer_table) as f:
        table = list(csv.DictReader(f))
    block_of = {n.name: table[i]["block"] for i, n in enumerate(original.graph.node)}

    # Assign original layers that were fused away to the optimised layer that absorbed them:
    # follow the removed layer's output downstream until reaching a tensor the optimised graph produces.
    opt_names = {n.name for n in opt.graph.node}
    producer = {o: n.name for n in opt.graph.node for o in n.output}
    consumers = defaultdict(list)
    for n in original.graph.node:
        for i in n.input:
            consumers[i].append(n)
    absorbed = defaultdict(list)
    for n in original.graph.node:
        if n.name in opt_names:
            continue
        t = n.output[0]
        while t not in producer:
            t = consumers[t][0].output[0]
        absorbed[producer[t]].append(n.name)

    # Execution order: the profiler's order if available, else the order in the optimised file.
    order = [n.name for n in opt.graph.node]
    prof = os.path.join(run_dir, "per_layer.csv")
    if os.path.exists(prof):
        with open(prof) as f:
            prof_order = [r["layer"] for r in sorted(csv.DictReader(f), key=lambda r: int(r["order"]))]
        if sorted(prof_order) == sorted(order):
            order = prof_order
    nodes = {n.name: n for n in opt.graph.node}

    # Activation lifetimes: [index of writer, index of last reader]; the graph input exists from the start.
    graph_inputs = {i.name for i in opt.graph.input} - set(weights)
    graph_outputs = {o.name for o in opt.graph.output}
    first, last = {t: 0 for t in graph_inputs}, {}
    for idx, name in enumerate(order):
        n = nodes[name]
        for t in n.input:
            if t and t not in weights:
                last[t] = idx
        for t in n.output:
            first[t] = idx
            last.setdefault(t, idx)
    for t in graph_outputs:
        last[t] = len(order) - 1

    rows = []
    for idx, name in enumerate(order):
        n = nodes[name]
        act_in = [t for t in n.input if t and t not in weights]
        w_in = [t for t in n.input if t in weights]
        macs, flops = count_flops(n, shapes, weights)
        in_b = sum(nbytes(shapes, t) for t in act_in)
        out_b = sum(nbytes(shapes, t) for t in n.output)
        w_b = sum(weights[t].nbytes for t in w_in)
        live = [t for t in first if first[t] <= idx <= last[t]]
        moved = in_b + w_b + out_b
        rows.append({
            "order": idx, "layer": name, "op_type": n.op_type, "block": block_of[name],
            "group": GROUP[block_of[name]],
            "fused_from": "+".join([name] + absorbed.get(name, [])),
            "in_shape": "x".join(map(str, shapes[act_in[0]][0][1:])),
            "out_shape": "x".join(map(str, shapes[n.output[0]][0][1:])),
            "MACs": macs, "FLOPs": flops,
            "weight_MiB": round(w_b / MIB, 4), "act_in_MiB": round(in_b / MIB, 3),
            "act_out_MiB": round(out_b / MIB, 3), "bytes_moved_MiB": round(moved / MIB, 3),
            "intensity_FLOP_per_byte": round(flops / moved, 3),
            "live_act_MiB": round(sum(nbytes(shapes, t) for t in live) / MIB, 3),
            "live_tensors": len(live),
        })

    # Per block.
    blocks = []
    tot_macs, tot_flops = sum(r["MACs"] for r in rows), sum(r["FLOPs"] for r in rows)
    for g in GROUP_ORDER:
        rs = [r for r in rows if r["group"] == g]
        blocks.append({
            "group": g, "layers": len(rs), "MACs": sum(r["MACs"] for r in rs), "FLOPs": sum(r["FLOPs"] for r in rs),
            "FLOPs_share_pct": round(100 * sum(r["FLOPs"] for r in rs) / tot_flops, 3),
            "weight_MiB": round(sum(r["weight_MiB"] for r in rs), 3),
            "bytes_moved_MiB": round(sum(r["bytes_moved_MiB"] for r in rs), 2),
            "max_live_act_MiB": max(r["live_act_MiB"] for r in rs),
        })

    peak = max(rows, key=lambda r: r["live_act_MiB"])
    pidx = peak["order"]
    at_peak = [{"tensor": t, "shape": "x".join(map(str, shapes[t][0])), "MiB": round(nbytes(shapes, t) / MIB, 2),
                "written_by": order[first[t]] if t not in graph_inputs else "graph input", "last_read_by": order[last[t]]}
               for t in first if first[t] <= pidx <= last[t]]
    table_macs = sum(int(r["MACs"]) for r in table)
    weights_mib = sum(w.nbytes for w in weights.values()) / MIB
    summary = {
        "run_dir": os.path.relpath(run_dir, HERE), "layers": len(rows),
        "total_MACs_G": round(tot_macs / 1e9, 3), "total_FLOPs_G": round(tot_flops / 1e9, 3),
        "check_layer_table_MACs_G": round(table_macs / 1e9, 3),
        "weights_MiB": round(weights_mib, 2),
        "activation_peak": {"layer": peak["layer"], "op_type": peak["op_type"], "group": peak["group"],
                            "live_act_MiB": peak["live_act_MiB"], "tensors": at_peak},
        "peak_estimate_MiB": {"activations": peak["live_act_MiB"], "weights": round(weights_mib, 2),
                              "total": round(peak["live_act_MiB"] + weights_mib, 2)},
        "largest_activation": max(({"tensor": t, "MiB": round(nbytes(shapes, t) / MIB, 2)} for t in first),
                                  key=lambda d: d["MiB"]),
        "fused": {r["layer"]: r["fused_from"] for r in rows if "+" in r["fused_from"]},
    }

    for path, data in (("layer_stats.csv", rows), ("block_stats.csv", blocks)):
        with open(os.path.join(run_dir, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(run_dir, "layer_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{len(rows)} layers | {summary['total_MACs_G']} GMACs (layer_table: {summary['check_layer_table_MACs_G']}) | "
          f"{summary['total_FLOPs_G']} GFLOPs | weights {summary['weights_MiB']} MiB")
    print(f"activation peak {peak['live_act_MiB']} MiB at {peak['layer']} ({peak['op_type']}, {peak['group']}); "
          f"with weights {summary['peak_estimate_MiB']['total']} MiB")
    for t in at_peak:
        print(f"  live: {t['tensor']:<28} {t['shape']:<20} {t['MiB']:8.2f} MiB  ({t['written_by']} -> {t['last_read_by']})")
    print(f"{'group':<9} {'GFLOPs':>8} {'share':>7} {'weights MiB':>12} {'moved MiB':>10} {'max live MiB':>13}")
    for b in blocks:
        print(f"{b['group']:<9} {b['FLOPs']/1e9:8.3f} {b['FLOPs_share_pct']:6.2f}% {b['weight_MiB']:12.2f} "
              f"{b['bytes_moved_MiB']:10.1f} {b['max_live_act_MiB']:13.1f}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
