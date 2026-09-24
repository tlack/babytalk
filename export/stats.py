"""Params, MACs and a naive activation-memory estimate for the static ONNX graph.

  uv run stats.py [path.onnx]

MACs: Conv = out_elems * (Cin/groups) * K; ReduceMean/Mul/Add/Relu/Sigmoid counted as
element ops (reported separately). Activation peak: greedy liveness over the topological
order (a tensor lives from its producer until its last consumer), int8 = 1 byte/elem;
ESP-DL's own planner may do better (in-place ops) or worse (alignment, NWC scratch).
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import shape_inference

from citrinet import DATA


def main(path):
    g = shape_inference.infer_shapes(onnx.load(path)).graph
    shapes = {}
    for v in list(g.input) + list(g.value_info) + list(g.output):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    inits = {t.name: list(t.dims) for t in g.initializer}
    params = sum(int(np.prod(d)) for d in inits.values())
    macs, elem = defaultdict(int), 0
    for n in g.node:
        if n.op_type == "Conv":
            w = inits[n.input[1]]
            out = shapes[n.output[0]]
            groups = next((a.i for a in n.attribute if a.name == "group"), 1)
            kind = "depthwise" if groups > 1 else ("pointwise/1x1" if w[2] == 1 else "full")
            macs[kind] += int(np.prod(out)) * w[1] * w[2]
        elif n.op_type in ("ReduceMean", "Mul", "Add", "Relu", "Sigmoid"):
            elem += int(np.prod(shapes[n.input[0]]))
    T = shapes[g.input[0].name][-1]
    secs = T / 100
    total = sum(macs.values())
    print(f"graph: {path}")
    print(f"input {shapes[g.input[0].name]} = {secs:.0f} s of audio; output {shapes[g.output[0].name]}")
    print(f"params: {params:,}")
    print(f"conv MACs per window: {total / 1e9:.3f} G  -> {total / secs / 1e6:.1f} M MAC per second of audio")
    for k, v in macs.items():
        print(f"   {k:14s} {v / 1e9:.3f} G ({100 * v / total:.1f}%)")
    print(f"element-wise ops per window: {elem / 1e6:.1f} M")

    # liveness
    last_use = {}
    for i, n in enumerate(g.node):
        for x in n.input:
            last_use[x] = i
    live, peak, peak_at = {g.input[0].name: int(np.prod(shapes[g.input[0].name]))}, 0, None
    for i, n in enumerate(g.node):
        for o in n.output:
            if o in shapes:
                live[o] = int(np.prod(shapes[o]))
        cur = sum(live.values())
        if cur > peak:
            peak, peak_at = cur, n.name
        for x in n.input:
            if last_use.get(x) == i and x in live:
                del live[x]
    print(f"naive activation peak: {peak / 1024:.0f} KB at int8 ({2 * peak / 1024:.0f} KB int16), at {peak_at}")
    biggest = max(int(np.prod(s)) for s in shapes.values())
    print(f"largest single activation: {biggest / 1024:.0f} KB int8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DATA / "models" / "citrinet256_static1600.onnx"))
