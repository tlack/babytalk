"""Pack an mmrt image whose 1x1-conv weights use at most 16 int8 levels per output channel
(int4_gptq.py's output) into the 4-bit codebook format (MMRT_W_CB4, mmrt/mmrt.h), and
check that decoding gives back exactly the same int8 weights.

  uv run mmrt_cb4.py data/models/mmrt/citrinet256_cb4_gptq16.mmrt [-o out.mmrt]

Ops pack only where it saves bytes (256 + 8*C < 16*C, i.e. C > 32); the rest stay int8.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from mmrt_export import HDR, OP, VERSION, align16

W_INT8, W_CB4 = 0, 1


def pack_group(w):
    """w [C][16] int8 (<= 16 levels per lane) -> 256-byte table + C*8 index bytes."""
    C = w.shape[0]
    tab = np.zeros((16, 16), np.int8)
    idx = np.zeros((C, 16), np.uint8)
    for lane in range(16):
        lv = np.unique(w[:, lane])
        assert len(lv) <= 16, f"lane has {len(lv)} levels"
        tab[lane, : len(lv)] = lv
        tab[lane, len(lv):] = lv[-1]
        idx[:, lane] = np.searchsorted(lv, w[:, lane])
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    return tab.tobytes() + packed.tobytes()


def unpack_group(b, C):
    tab = np.frombuffer(b, np.int8, 256).reshape(16, 16)
    p = np.frombuffer(b, np.uint8, C * 8, 256).reshape(C, 8)
    idx = np.empty((C, 16), np.uint8)
    idx[:, 0::2], idx[:, 1::2] = p & 15, p >> 4
    return tab[np.arange(16)[None], idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out) if a.out else src.with_name(src.stem.replace("_cb4", "") + "_cb4packed.mmrt")
    img = src.read_bytes()
    h = list(HDR.unpack_from(img))
    n_t, n_ops, t_off, o_off, blob_off = h[2], h[3], h[6], h[7], h[8]
    ch = [struct.unpack_from("<H", img, t_off + 4 * i)[0] for i in range(n_t)]
    blob = img[blob_off:]
    new, seen = bytearray(), {}

    def add(data):
        off = len(new)
        new.extend(data + b"\0" * (align16(len(data)) - len(data)))
        return off

    ops = []
    packed_ops = saved = 0
    for i in range(n_ops):
        f = list(OP.unpack_from(img, o_off + 28 * i))
        kind, K, in0, out_t, w_off, b_off = f[0], f[2], f[9], f[11], f[12], f[13]
        C, N = ch[in0], ch[out_t]
        wfmt = W_INT8
        if w_off != 0xFFFFFFFF:
            size = {1: C * K, 2: C * N, 4: 256}[kind]
            data = blob[w_off: w_off + size]
            if kind == 2 and 256 + 8 * C < 16 * C:
                w = np.frombuffer(data, np.int8).reshape(N // 16, C, 16)
                data = b"".join(pack_group(w[g]) for g in range(N // 16))
                dec = np.stack([unpack_group(data[g * (256 + 8 * C):], C) for g in range(N // 16)])
                assert np.array_equal(dec, w), f"op {i}: decode mismatch"
                wfmt = W_CB4
                packed_ops += 1
                saved += size - len(data)
            key = ("w", w_off, wfmt)
            if key not in seen:  # LUTs are shared between ops
                seen[key] = add(data)
            f[12] = seen[key]
        if b_off != 0xFFFFFFFF:
            f[13] = add(blob[b_off: b_off + 4 * N])
        ops.append((f, wfmt))
    raw = bytearray(img[:blob_off])
    for i, (f, wfmt) in enumerate(ops):
        OP.pack_into(raw, o_off + 28 * i, *f)
        raw[o_off + 28 * i + 9] = wfmt
    h[1], h[9] = VERSION, len(new)
    HDR.pack_into(raw, 0, *h)
    out.write_bytes(bytes(raw) + bytes(new))
    print(f"-> {out}: {packed_ops} ops cb4, {len(img) / 1e6:.2f} -> {(len(raw) + len(new)) / 1e6:.2f} MB "
          f"(saved {saved / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
