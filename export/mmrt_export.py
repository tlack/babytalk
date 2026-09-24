"""Export the deployed ESP-DL graph (mmrt_quant.py's .info) to an mmrt model file.

  uv run mmrt_export.py        -> data/models/mmrt/citrinet256_int8.mmrt

Layout and field meanings: mmrt/mmrt.h. Weights keep ESP-DL's verified layouts
([C/16][K][16] depthwise, [N/16][C][16] 1x1). The 257-way decoder (ESP-DL's
"UNALIGNED" layout: 16 aligned groups + 1 trailing channel) is padded to 272 outputs
with zero weights and a very negative bias, so every 1x1 conv takes the aligned
kernel and the padding can never win the CTC argmax.
"""
import struct

import numpy as np

from mmrt_info import MMRT, parse

OUT = MMRT / "citrinet256_int8.mmrt"
MAGIC, VERSION = 0x54524D4D, 1
KIND = {"dwconv": 1, "conv1x1": 2, "mean": 3, "lut": 4, "mul": 5, "add": 6}
NONE16, NONE32 = 0xFFFF, 0xFFFFFFFF
HDR = struct.Struct("<16I")          # mmrt_header_t: 11 fields + 5 reserved
TENSOR = struct.Struct("<Hbx")       # channels, exp, pad
OP = struct.Struct("<BBBBbbbbb3xHHHxxII")  # mmrt_op_t (28 bytes)
assert TENSOR.size == 4 and OP.size == 28 and HDR.size == 64  # = sizeof() in mmrt.h


def align16(n):
    return (n + 15) & ~15


class Blob:
    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr):
        off = self.size
        data = np.ascontiguousarray(arr).tobytes()
        self.parts.append(data + b"\0" * (align16(len(data)) - len(data)))
        self.size += align16(len(data))
        return off

    def bytes(self):
        return b"".join(self.parts)


def pad_decoder(w, bias, C):
    """UNALIGNED [16 groups][C][16] + [1][C] -> aligned [17 groups][C][16], N 257 -> 272."""
    v = w.value.astype(np.int8)
    K, Cw, N = w.shape
    na = N // 16 * 16
    aligned = v[: na * C].reshape(na // 16, C, 16)
    tail = np.zeros((C, 16), np.int8)
    tail[:, : N - na] = v[na * C:].reshape(N - na, C).T
    w_pad = np.concatenate([aligned, tail[None]], axis=0)
    b_pad = np.full(align16(N), -(1 << 24), np.int32)  # saturates to -128 at any shift
    assert not bias.value[N:].any()  # ESP-DL pads the stored bias (260 for 257)
    b_pad[:N] = bias.value[:N]
    return w_pad, b_pad, align16(N), N


def main():
    g = parse()
    act_names = [g.nodes[0].inputs[0]] + [n.out for n in g.nodes]
    tid = {name: i for i, name in enumerate(act_names)}
    channels = {}
    for name in act_names:
        channels[name] = g.tensors[name].shape[-1]
    blob, ops = Blob(), []
    lut_off = {}
    out_valid = None
    for n in g.nodes:
        e = lambda name: g.tensors[name].exp
        e_out = e(n.out)
        rec = dict(kind=0, relu=0, K=0, stride=1, pad=0, shift=0, e_a=0, e_b=0, e_out=e_out,
                   in0=tid[n.inputs[0]], in1=NONE16, out=tid[n.out], w_off=NONE32, b_off=NONE32)
        if n.op == "Conv":
            w = g.tensors[n.inputs[1]]
            rec.update(K=w.shape[0], stride=n.attrs["strides"][0], pad=n.attrs["pads"][0],
                       relu=int(n.attrs["activation"] == "Relu"), shift=e_out - e(n.inputs[0]) - w.exp)
            assert 0 <= rec["shift"] < 32, (n.out, rec["shift"])
            if n.attrs["group"] > 1:
                rec.update(kind=KIND["dwconv"], w_off=blob.add(w.value.astype(np.int8)))
            else:
                b = g.tensors[n.inputs[2]] if len(n.inputs) > 2 else None  # SE convs have none
                assert b is None or b.exp == e(n.inputs[0]) + w.exp
                if w.layout.endswith("UNALIGNED"):
                    wv, bv, N, out_valid = pad_decoder(w, b, w.shape[1])
                    channels[n.out] = N
                else:
                    wv, bv = w.value.astype(np.int8), (b.value.astype(np.int32) if b else None)
                rec.update(kind=KIND["conv1x1"], w_off=blob.add(wv), b_off=blob.add(bv) if bv is not None else NONE32)
        elif n.op == "ReduceMean":
            rec.update(kind=KIND["mean"], e_a=e(n.inputs[0]))
        elif n.op in ("Sigmoid", "Relu"):
            name = n.attrs["lut"]
            if name not in lut_off:
                lut_off[name] = blob.add(g.tensors[name].value.astype(np.int8))
            rec.update(kind=KIND["lut"], w_off=lut_off[name])
        elif n.op == "Mul":
            a, b = n.inputs[:2]
            if g.tensors[a].shape[-2] == 1:  # vector first: swap so in1 is the [1][C] scale
                a, b = b, a
            rec.update(kind=KIND["mul"], in0=tid[a], in1=tid[b], e_a=e(a), e_b=e(b))
        elif n.op == "Add":
            a, b = n.inputs[:2]
            rec.update(kind=KIND["add"], in0=tid[a], in1=tid[b], e_a=e(a), e_b=e(b))
        else:
            raise NotImplementedError(n.op)
        ops.append(rec)

    tensors = b"".join(TENSOR.pack(channels[name], g.tensors[name].exp) for name in act_names)
    ops_b = b"".join(OP.pack(o["kind"], o["relu"], o["K"], o["stride"], o["pad"], o["shift"], o["e_a"],
                             o["e_b"], o["e_out"], o["in0"], o["in1"], o["out"], o["w_off"], o["b_off"])
                     for o in ops)
    tensors_off = align16(HDR.size)
    ops_off = align16(tensors_off + len(tensors))
    blob_off = align16(ops_off + len(ops_b))
    hdr = HDR.pack(MAGIC, VERSION, len(act_names), len(ops), tid[act_names[0]], tid[g.nodes[-1].out],
                   tensors_off, ops_off, blob_off, blob.size, out_valid or channels[g.nodes[-1].out], 0, 0, 0, 0, 0)
    img = bytearray(blob_off + blob.size)
    img[: len(hdr)] = hdr
    img[tensors_off: tensors_off + len(tensors)] = tensors
    img[ops_off: ops_off + len(ops_b)] = ops_b
    img[blob_off:] = blob.bytes()
    OUT.write_bytes(img)
    print(f"-> {OUT}: {len(ops)} ops, {len(act_names)} tensors, blob {blob.size / 1e6:.2f} MB, "
          f"file {len(img) / 1e6:.2f} MB; output {channels[g.nodes[-1].out]} ch ({out_valid} valid)")


if __name__ == "__main__":
    main()
