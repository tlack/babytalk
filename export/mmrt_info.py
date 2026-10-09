"""Parse ESP-PPQ's .info dump of a deployed ESP-DL graph (mmrt_quant.py output).

The .info file is the ground truth of what runs on the chip: every node with its
attributes, every initializer's int values (in ESP-DL's kernel layout) and exponent,
and every activation's shape and exponent. mmrt_export.py builds the runtime model
from this.

  uv run mmrt_info.py      # parse + verify weight layouts against the ONNX floats
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from citrinet import MODELS

MMRT = MODELS / "mmrt"
INFO = MMRT / "citrinet256_int8.info"
ONNX = MMRT / "citrinet256_static1600_cle.onnx"  # the (simplified) graph ESP-PPQ quantized


@dataclass
class Tensor:
    name: str
    shape: list[int]
    exp: int
    layout: str = ""
    value: np.ndarray | None = None  # initializers only, flat, in ESP-DL layout


@dataclass
class Node:
    op: str
    out: str
    inputs: list[str]
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    inputs: dict[str, Tensor]
    nodes: list[Node]
    tensors: dict[str, Tensor]  # activations + initializers


_NODE = re.compile(r"^\s*%(\S+) = (\w+)\[(.*)\]\((.*)\)\s*$")
_VALINFO = re.compile(r"^%(\S+)\[(\w+), ([\dx]+)\], exponents: \[(-?\d+)\]")
_INIT = re.compile(r"^%(\S+), shape: \[([\d, ]*)\], exponents: \[(-?\d+)\],(?: docString: b'layout ==> ([^']*)',)?")


def _attrs(s: str) -> dict:
    out = {}
    for m in re.finditer(r"(\w+) = (\[[^\]]*\]|'[^']*'|-?\d+)", s):
        out[m.group(1)] = ast.literal_eval(m.group(2))
    return out


def parse(path: Path = INFO) -> Graph:
    lines = path.read_text().splitlines()
    inputs, nodes, tensors = {}, [], {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _NODE.match(line)
        if m:
            nodes.append(Node(m.group(2), m.group(1), [a.strip().lstrip("%") for a in m.group(4).split(",")],
                              _attrs(m.group(3))))
            i += 1
            continue
        m = _INIT.match(line)
        if m and i + 1 < len(lines) and lines[i + 1].startswith("value: array("):
            # value: array([...], dtype=int8) may span many lines
            j, buf = i + 1, []
            while True:
                buf.append(lines[j])
                if "dtype=" in lines[j] or lines[j].rstrip().endswith("])"):
                    break
                j += 1
            text = " ".join(buf)
            body = text[text.index("[") : text.rindex("]") + 1]
            dt = re.search(r"dtype=(\w+)", text)
            arr = np.array(ast.literal_eval(body), dtype=dt.group(1) if dt else np.int64).ravel()
            shape = [int(x) for x in m.group(2).split(",") if x.strip()]
            tensors[m.group(1)] = Tensor(m.group(1), shape, int(m.group(3)), m.group(4) or "", arr)
            i = j + 1
            continue
        m = _VALINFO.match(line)
        if m:
            name = m.group(1)
            if name not in tensors:
                tensors[name] = Tensor(name, [int(x) for x in m.group(3).split("x")], int(m.group(4)))
        i += 1
    # graph inputs: "%feats[INT8, 1x1600x80], exponents: [-5]" in the header
    for line in lines[:10]:
        m = re.match(r"^\s*%(\S+)\[(\w+), ([\dx]+)\], exponents: \[(-?\d+)\]", line)
        if m:
            inputs[m.group(1)] = Tensor(m.group(1), [int(x) for x in m.group(3).split("x")], int(m.group(4)))
    tensors.update(inputs)
    return Graph(inputs, nodes, tensors)


def unpermute_weight(t: Tensor, group: int) -> np.ndarray:
    """ESP-DL '(N/16)WC16' conv weight -> logical int8 [N_out, K, C_in] (C_in = 1 for depthwise).

    t.shape is [K, C, N] ([K, C, 1] for depthwise, where the channel axis is C). Storage:
    for each 16-output group g, for each tap k, for each input channel c: 16 outputs.
    Depthwise: for each 16-channel group, for each tap k: 16 channel weights."""
    K, C, N = t.shape
    v = t.value
    if group > 1:  # depthwise: N == 1, channels on the C axis
        assert N == 1 and C % 16 == 0
        return v.reshape(C // 16, K, 16).transpose(0, 2, 1).reshape(C, K, 1)
    if t.layout.endswith("UNALIGNED"):
        raise NotImplementedError
    assert N % 16 == 0, t.name
    return v.reshape(N // 16, K, C, 16).transpose(0, 3, 1, 2).reshape(N, K, C)


def verify_layouts(g: Graph, verbose=True) -> None:
    """Logical weights from the .info must equal round(onnx_float / 2^exp)."""
    import onnx
    from onnx import numpy_helper

    onnx_w = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(ONNX)).graph.initializer}
    checked = worst = 0
    for n in g.nodes:
        if n.op != "Conv":
            continue
        t = g.tensors[n.inputs[1]]
        if t.layout.endswith("UNALIGNED"):
            continue
        logical = unpermute_weight(t, n.attrs.get("group", 1))  # [N, K, C]
        if t.name not in onnx_w:
            if verbose:
                print(f"  (no ONNX float for {t.name}: folded/renamed by ESP-PPQ)")
            continue
        w = onnx_w[t.name]  # ONNX Conv: [N, C/g, K]
        ref = np.clip(np.floor(w / 2.0 ** t.exp + 0.5), -128, 127).transpose(0, 2, 1)
        diff = np.abs(ref - logical)
        worst = max(worst, diff.max())
        assert diff.max() <= 1 and (diff > 0).mean() < 1e-3, (t.name, diff.max(), (diff > 0).mean())
        checked += 1
    print(f"weight layouts verified against ONNX floats: {checked} convs, worst |diff| {worst} "
          f"(1 = a rounding tie)")


if __name__ == "__main__":
    import collections

    g = parse()
    inits = [t for t in g.tensors.values() if t.value is not None]
    print(f"{len(g.nodes)} nodes, {len(inits)} initializers, "
          f"{len(g.tensors) - len(inits) - len(g.inputs)} activations, inputs {list(g.inputs)}")
    print(collections.Counter(n.op for n in g.nodes))
    print(collections.Counter((n.op, n.attrs.get("activation", "")) for n in g.nodes if n.op == "Conv"))
    verify_layouts(g)
