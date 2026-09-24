"""Check mmrt ops against ESP-PPQ's simulation, one graph node at a time.

Each node gets ESP-PPQ's own input tensors (data/models/mmrt/ref/<clip>.npz) and its
output is compared with ESP-PPQ's output tensor: every op is tested in isolation, no
error carried from earlier ops.

  uv run mmrt_check_ops.py [clip ...]        # default: all ref clips
  uv run mmrt_check_ops.py --lib path.so     # check another build of the same C API
"""
import argparse
import collections
import ctypes
import subprocess
from pathlib import Path

import numpy as np

from mmrt_info import MMRT, parse

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "mmrt" / "mmrt_ref.c"
LIB = ROOT / "mmrt" / "build" / "libmmrt_ref.so"

P8 = ctypes.POINTER(ctypes.c_int8)
P32 = ctypes.POINTER(ctypes.c_int32)


def build():
    LIB.parent.mkdir(exist_ok=True)
    subprocess.check_call(["gcc", "-O2", "-Wall", "-shared", "-fPIC", "-o", str(LIB), str(SRC)])
    return str(LIB)


def p8(a):
    return np.ascontiguousarray(a, dtype=np.int8).ctypes.data_as(P8)


def p32(a):
    return np.ascontiguousarray(a, dtype=np.int32).ctypes.data_as(P32)


class Ops:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        I = ctypes.c_int
        self.lib.mmrt_dwconv_ref.argtypes = [P8, I, I, P8, I, I, I, I, I, P8, I]
        self.lib.mmrt_conv1x1_ref.argtypes = [P8, I, I, P8, P32, I, I, I, I, P8, I]
        self.lib.mmrt_mean_ref.argtypes = [P8, I, I, I, I, P8]
        self.lib.mmrt_lut_ref.argtypes = [P8, I, P8, P8]
        self.lib.mmrt_mul_bcast_ref.argtypes = [P8, I, I, P8, I, I, I, P8]
        self.lib.mmrt_add_ref.argtypes = [P8, P8, I, I, I, I, P8]

    def max_acc(self):
        return ctypes.c_int32.in_dll(self.lib, "mmrt_ref_max_acc").value


def conv_out_len(T, K, stride, pad):
    return (T + 2 * pad - K) // stride + 1


def run_node(ops, g, n, act, exps):
    """Run one node on reference inputs; returns (op kind, output array)."""
    T = lambda name: act[name].shape[0]
    x = act[n.inputs[0]]
    e = lambda name: g.tensors[name].exp
    e_out = e(n.out)
    if n.op == "Conv":
        w = g.tensors[n.inputs[1]]
        K = w.shape[0]
        stride, pad, group = n.attrs["strides"][0], n.attrs["pads"][0], n.attrs["group"]
        relu = n.attrs["activation"] == "Relu"
        shift = e_out - e(n.inputs[0]) - w.exp
        Tin, C = x.shape
        Tout = conv_out_len(Tin, K, stride, pad)
        if group > 1:
            y = np.zeros((Tout, C), np.int8)
            ops.lib.mmrt_dwconv_ref(p8(x), Tin, C, p8(w.value), K, stride, pad, shift, relu, p8(y), Tout)
            return "dwconv", y
        if w.layout.endswith("UNALIGNED"):
            return "conv1x1-unaligned (skipped)", None
        assert K == 1
        N = w.shape[2]
        bias = None
        if len(n.inputs) > 2:
            b = g.tensors[n.inputs[2]]
            assert b.exp == e(n.inputs[0]) + w.exp, (n.out, b.exp, e(n.inputs[0]), w.exp)
            bias = b.value
        y = np.zeros((Tout, N), np.int8)
        ops.lib.mmrt_conv1x1_ref(p8(x), Tin, C, p8(w.value), p32(bias) if bias is not None else None, N,
                                 stride, shift, relu, p8(y), Tout)
        return "conv1x1" + ("+relu" if relu else ""), y
    if n.op == "ReduceMean":
        Tin, C = x.shape
        y = np.zeros((1, C), np.int8)
        ops.lib.mmrt_mean_ref(p8(x), Tin, C, e(n.inputs[0]), e_out, p8(y))
        return "mean", y
    if n.op in ("Sigmoid", "Relu"):
        lut = g.tensors[n.attrs["lut"]].value.astype(np.int8)
        y = np.zeros_like(x)
        ops.lib.mmrt_lut_ref(p8(x), x.size, p8(lut), p8(y))
        return n.op.lower() + "-lut", y
    if n.op == "Mul":
        a, b = act[n.inputs[0]], act[n.inputs[1]]
        (x, ex), (s, es) = sorted([(a, e(n.inputs[0])), (b, e(n.inputs[1]))], key=lambda p: -p[0].shape[0])
        Tx, C = x.shape
        y = np.zeros_like(x)
        ops.lib.mmrt_mul_bcast_ref(p8(x), Tx, C, p8(s.reshape(-1)), ex, es, e_out, p8(y))
        return "mul-bcast", y
    if n.op == "Add":
        a, b = act[n.inputs[0]], act[n.inputs[1]]
        assert a.shape == b.shape
        y = np.zeros_like(a)
        ops.lib.mmrt_add_ref(p8(a), p8(b), a.size, e(n.inputs[0]), e(n.inputs[1]), e_out, p8(y))
        return "add", y
    raise NotImplementedError(n.op)


def check_clip(ops, g, clip):
    ref = np.load(MMRT / "ref" / f"{clip}.npz", allow_pickle=True)
    act = {k: ref[k] for k in ref.files if not k.startswith("__")}
    exps = dict(ref["__exponents__"])
    for name, ex in exps.items():  # the dump's exponents must agree with the deployed graph
        if name in g.tensors:
            assert g.tensors[name].exp == ex, (name, g.tensors[name].exp, ex)
    stats = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # ops, exact ops, elems, bad elems, max diff
    worst = []
    for n in g.nodes:
        kind, y = run_node(ops, g, n, act, exps)
        s = stats[kind]
        s[0] += 1
        if y is None:
            continue
        if n.out not in act:  # fused away in the simulation (e.g. Add before a Relu LUT): feed
            act[n.out] = y   # ours forward; the next op's comparison checks both together
            s[0] -= 1
            stats[kind + " (checked via next op)"][0] += 1
            continue
        r = act[n.out].reshape(y.shape)
        d = np.abs(y.astype(np.int32) - r.astype(np.int32))
        s[1] += int(d.max() == 0)
        s[2] += d.size
        s[3] += int((d > 0).sum())
        s[4] = max(s[4], int(d.max()))
        if d.max() > 0:
            worst.append((int((d > 0).sum()), int(d.max()), n.out))
    print(f"== {clip}  (T_in={act['feats'].shape[0]})")
    for kind, (nops, exact, elems, bad, md) in sorted(stats.items()):
        tail = f"{bad}/{elems} elems differ, max {md}" if elems else ""
        print(f"  {kind:28s} {exact:3d}/{nops:3d} bit-exact  {tail}")
    for nb, md, name in sorted(worst, reverse=True)[:5]:
        print(f"    worst: {name}: {nb} elems, max diff {md}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", default=["me-hello-world", "me-we-the-people", "1089-134686-0000"])
    ap.add_argument("--lib", default=None)
    a = ap.parse_args()
    ops = Ops(a.lib or build())
    g = parse()
    for clip in a.clips:
        check_clip(ops, g, clip)
    print(f"largest |conv accumulator|: {ops.max_acc()} (QACC int8 lanes wrap at 2^19 = {2**19})")


if __name__ == "__main__":
    main()
