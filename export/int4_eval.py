"""4-bit weights experiment: rewrite an mmrt image's 1x1-conv weights so every output
channel uses at most 16 distinct int8 values (a per-channel codebook: what a packed
4-bit format with a 16-entry int8 table per channel decodes to), then score WER with
the host C executor (bit-exact with the board).

  uv run int4_eval.py                          # int8 baseline + codebook variants
  uv run int4_eval.py --variants kmeans --every 4 --recordings

Variants:
  int8       the image as is
  kmeans     per output channel: 1-D k-means (16 levels) on its int8 weights
  uniform    per output channel: symmetric uniform int4 (levels s*[-8..7], s = max|w|/7.5)
  kmeans_io  kmeans, but the first 1x1 and the CTC decoder stay int8
Depthwise weights (~255KB) stay int8 in every variant.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import struct
import time

import numpy as np

from citrinet import DATA, ctc_greedy, load_model, load_split, read_audio
from evalwer import fixed_windows, out_len, score
from mmrt_check_model import ALLOC, FREE, LIB, MODEL, build

OUT = DATA / "models" / "mmrt"
RES = DATA / "results"


# ---------------------------------------------------------------- image rewrite
def conv1x1_ops(img: bytes):
    """-> [(op index, w_off, C_in, N_out)] for every 1x1 conv in the image."""
    h = struct.unpack_from("<16I", img)
    n_t, n_ops, t_off, o_off, blob_off = h[2], h[3], h[6], h[7], h[8]
    ch = [struct.unpack_from("<H", img, t_off + 4 * i)[0] for i in range(n_t)]
    out = []
    for i in range(n_ops):
        kind, *_ = struct.unpack_from("<B", img, o_off + 28 * i)
        in0, in1, o = struct.unpack_from("<HHH", img, o_off + 28 * i + 12)
        w_off, = struct.unpack_from("<I", img, o_off + 28 * i + 20)
        if kind == 2:
            out.append((i, blob_off + w_off, ch[in0], ch[o]))
    return out


def kmeans_1d(v: np.ndarray, k: int = 16, iters: int = 30) -> np.ndarray:
    """Integer levels (<= k) minimizing squared error for int8 values v."""
    u = np.unique(v)
    if len(u) <= k:
        return u.astype(np.float64)
    c = np.quantile(v.astype(np.float64), (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        a = np.abs(v[:, None] - c[None]).argmin(1)
        nc = np.array([v[a == j].mean() if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return np.unique(np.clip(np.round(c), -128, 127))


def quantize_channel(v: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        s = max(np.abs(v).max() / 7.5, 1e-9)
        levels = np.unique(np.clip(np.round(np.arange(-8, 8) * s), -128, 127))
    else:
        levels = kmeans_1d(v)
    return levels[np.abs(v[:, None].astype(np.float64) - levels[None]).argmin(1)].astype(np.int8)


def rewrite(img: bytes, mode: str, keep_io: bool) -> tuple[bytes, dict]:
    b = bytearray(img)
    ops = conv1x1_ops(img)
    keep = {ops[0][0], ops[-1][0]} if keep_io else set()
    err = num = 0.0
    for i, off, C, N in ops:
        if i in keep:
            continue
        w = np.frombuffer(img, np.int8, C * N, off).reshape(N // 16, C, 16)
        q = np.empty_like(w)
        for g in range(N // 16):
            for lane in range(16):
                q[g, :, lane] = quantize_channel(w[g, :, lane], mode)
        err += float(((q.astype(np.float64) - w) ** 2).sum())
        num += float((w.astype(np.float64) ** 2).sum())
        b[off : off + C * N] = q.tobytes()
    return bytes(b), {"rel_weight_err": (err / num) ** 0.5, "int8_ops_kept": len(keep)}


# ---------------------------------------------------------------- WER with the C executor
_W = {}


def _init(path):
    lib = ctypes.CDLL(str(LIB))  # built once by the parent (build())
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    img = open(path, "rb").read()
    buf = ctypes.create_string_buffer(img, len(img))
    model = ctypes.create_string_buffer(256)
    assert lib.mmrt_open(model, buf, len(img)) == 0
    lib.mmrt_run.restype = ctypes.POINTER(ctypes.c_int8)
    lib.mmrt_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int), ALLOC, FREE]
    _, feat, vocab, _ = load_model()
    _W.update(lib=lib, buf=buf, model=model, feat=feat, vocab=vocab,
              alloc=ALLOC(lambda n: libc.malloc(n)), free=FREE(lambda p: libc.free(p)))


def _one(item):
    import torch

    uid, path, ref = item
    audio = read_audio(path)
    feat, lib = _W["feat"], _W["lib"]
    lm = feat.logmel(audio)[:, : feat.seq_len(audio.shape[0])]
    parts = []
    for x, v in fixed_windows(lm, fill="zero"):  # clips > 16 s: equal chunks, as on the board
        q = np.clip(np.round(x[:, :v].numpy().T * 32), -128, 127).astype(np.int8)  # input exp -5
        q = np.ascontiguousarray(q)
        T_out = ctypes.c_int()
        y = lib.mmrt_run(_W["model"], q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), v, ctypes.byref(T_out),
                         _W["alloc"], _W["free"])
        assert y
        lg = np.ctypeslib.as_array(y, shape=(T_out.value * 272,)).reshape(T_out.value, 272)[:, :257]
        parts.append(ctc_greedy(torch.from_numpy(lg.T.astype(np.float32)), _W["vocab"], out_len(v)))
    return uid, ref, " ".join(p for p in parts if p), audio.shape[0] / 16000


def wer(path, items, jobs):
    with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(str(path),)) as pool:
        return pool.map(_one, items, chunksize=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["int8", "kmeans", "uniform", "kmeans_io"])
    ap.add_argument("--every", type=int, default=4, help="every Nth test-clean utterance (4: 655)")
    ap.add_argument("--recordings", action="store_true", help="also score data/recordings (live voice)")
    ap.add_argument("-j", type=int, default=22)
    a = ap.parse_args()
    build()
    img = MODEL.read_bytes()
    sets = {"test-clean": load_split()[:: a.every]}
    if a.recordings:
        from recordings import load_recordings
        sets["recordings"] = load_recordings()
    for var in a.variants:
        t0 = time.time()
        info = {}
        path = MODEL
        if var != "int8":
            new, info = rewrite(img, "uniform" if var == "uniform" else "kmeans", var.endswith("_io"))
            path = OUT / f"citrinet256_cb4_{var}.mmrt"
            path.write_bytes(new)
        row = {"variant": var, **info}
        for name, items in sets.items():
            rows = wer(path, items, a.j)
            row[name] = round(score(rows), 3)
            if name == "recordings":
                mine = [r for r in rows if r[0].startswith("me-")]
                row["live_voice"] = round(score(mine), 2) if mine else None
            RES.mkdir(parents=True, exist_ok=True)
            with open(RES / f"int4_{var}_{name}.tsv", "w") as f:
                for uid, ref, hyp, dur in rows:
                    f.write(f"{uid}\t{dur:.2f}\t{ref}\t{hyp}\n")
        row["secs"] = round(time.time() - t0)
        print(json.dumps(row), flush=True)
        with open(RES / "int4_summary.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
