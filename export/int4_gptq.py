"""GPTQ for the 4-bit codebook format: quantize each 1x1 conv's weights one input channel
at a time to its output channel's 16-level int8 codebook, pushing the rounding error onto
the not-yet-quantized input channels through the inverse input Hessian H = sum x x^T
(input activations from the int8 C executor on dev-clean, which test-clean never sees).

  uv run int4_gptq.py [--calib 128] [--levels 16] [--every 4] [--recordings]

Writes data/models/mmrt/citrinet256_cb4_gptq.mmrt (int8-valued: the decode of the packed
format) and scores it like int4_eval.py.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import struct
import time

import numpy as np

import int4_eval as E
from citrinet import load_split, read_audio
from evalwer import fixed_windows, score

TRACE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                         ctypes.c_int)


def graph(img: bytes):
    """-> (input tensor id, {op: (kind, in0, out)}), channels per tensor."""
    h = struct.unpack_from("<16I", img)
    n_t, n_ops, t_off, o_off = h[2], h[3], h[6], h[7]
    ch = [struct.unpack_from("<H", img, t_off + 4 * i)[0] for i in range(n_t)]
    ops = {}
    for i in range(n_ops):
        kind, = struct.unpack_from("<B", img, o_off + 28 * i)
        in0, _, out = struct.unpack_from("<HHH", img, o_off + 28 * i + 12)
        ops[i] = (kind, in0, out)
    return h[4], ops, ch


# ---------------------------------------------------------------- calibration: H per 1x1 input
_C = {}


def _cinit(path, want):
    E._init(path)
    _C["want"] = want  # tensor id -> channels
    _C["H"] = {t: np.zeros((c, c), np.float64) for t, c in want.items()}
    _C["n"] = {t: 0 for t in want}
    _C["s"] = {t: np.zeros(c, np.float64) for t, c in want.items()}


def _acc(t, x):
    x = x.astype(np.float64)
    _C["H"][t] += x.T @ x
    _C["s"][t] += x.sum(0)
    _C["n"][t] += x.shape[0]


def _calib(items):
    ops, inp = _C["ops"], _C["input"]

    def on_op(i, op_ptr, out, T, C):
        t = ops[i][2]
        if t in _C["want"]:
            _acc(t, np.ctypeslib.as_array(out, shape=(T * C,)).reshape(T, C))

    cb = TRACE(on_op)
    lib = E._W["lib"]
    ctypes.c_void_p.in_dll(lib, "mmrt_trace").value = ctypes.cast(cb, ctypes.c_void_p).value
    feat = E._W["feat"]
    for _, path, _ in items:
        audio = read_audio(path)
        lm = feat.logmel(audio)[:, : feat.seq_len(audio.shape[0])]
        for x, v in fixed_windows(lm, fill="zero"):
            q = np.ascontiguousarray(np.clip(np.round(x[:, :v].numpy().T * 32), -128, 127).astype(np.int8))
            if inp in _C["want"]:
                _acc(inp, q)
            T_out = ctypes.c_int()
            assert lib.mmrt_run(E._W["model"], q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), v,
                                ctypes.byref(T_out), E._W["alloc"], E._W["free"])
    ctypes.c_void_p.in_dll(lib, "mmrt_trace").value = None
    return _C["H"], _C["n"], _C["s"]


def calibrate(img, n_clips, jobs):
    """-> {tensor id: (E[x x^T], E[x])} over n_clips dev-clean clips (cached on disk)."""
    cache = E.RES / f"int4_calib{n_clips}.npz"
    if cache.exists():
        z = np.load(cache)
        return {int(k[1:]): (z[k], z["m" + k[1:]]) for k in z.files if k[0] == "H"}
    inp, ops, ch = graph(img)
    want = {ops[o[0]][1]: ch[ops[o[0]][1]] for o in E.conv1x1_ops(img)}
    items = load_split("dev-clean")
    items = items[:: max(1, len(items) // n_clips)][:n_clips]
    chunks = [items[k::jobs] for k in range(jobs)]
    _C.update(ops=ops, input=inp)
    with mp.get_context("fork").Pool(jobs, initializer=_cinit, initargs=(str(E.MODEL), want)) as pool:
        parts = pool.map(_calib, chunks, chunksize=1)
    H = {t: sum(p[0][t] for p in parts) for t in want}
    n = {t: sum(p[1][t] for p in parts) for t in want}
    S = {t: sum(p[2][t] for p in parts) for t in want}
    out = {t: (H[t] / max(1, n[t]), S[t] / max(1, n[t])) for t in want}
    np.savez(cache, **{f"H{t}": v[0] for t, v in out.items()}, **{f"m{t}": v[1] for t, v in out.items()})
    return out


# ---------------------------------------------------------------- GPTQ with per-row codebooks
def codebook(v, k, weight=None):
    """k integer levels for int8 values v (weighted 1-D k-means, quantile init)."""
    u = np.unique(v)
    if len(u) <= k:
        return u.astype(np.float64)
    w = np.ones_like(v, np.float64) if weight is None else weight
    c = np.quantile(v.astype(np.float64), (np.arange(k) + 0.5) / k)
    for _ in range(40):
        a = np.abs(v[:, None] - c[None]).argmin(1)
        nc = c.copy()
        for j in range(k):
            m = a == j
            if m.any():
                nc[j] = (v[m] * w[m]).sum() / w[m].sum()
        if np.allclose(nc, c):
            break
        c = nc
    lv = np.unique(np.clip(np.round(c), -128, 127))
    return lv


def gptq(W, H, k, damp=0.01, act_order=True):
    """W [N][C] (int8 values), H [C][C] -> Q [N][C] int8, each row within its own k levels.
    act_order: quantize input channels in order of decreasing E[x^2] (GPTQ's act-order)."""
    if act_order:
        perm = np.argsort(-np.diag(H))
        Q = gptq(W[:, perm], H[perm][:, perm], k, damp, act_order=False)
        out = np.empty_like(Q)
        out[:, perm] = Q
        return out
    W = W.astype(np.float64).copy()
    N, C = W.shape
    d = np.diag(H).copy()
    dead = d == 0
    H = H.copy()
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * d.mean() * np.eye(C)
    # upper Cholesky factor of H^-1 (as in GPTQ)
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T
    books = [codebook(W[r].round(), k, d + 1e-9) for r in range(N)]
    L = np.full((N, k), np.nan)
    for r, b in enumerate(books):
        L[r, : len(b)] = b
        L[r, len(b):] = b[-1]
    Q = np.zeros_like(W)
    for i in range(C):
        w = W[:, i]
        q = L[np.arange(N), np.abs(w[:, None] - L).argmin(1)]
        Q[:, i] = q
        err = (w - q) / Hinv[i, i]
        W[:, i + 1:] -= err[:, None] * Hinv[i, i + 1:][None]
    return Q.astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=128, help="dev-clean clips for H")
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--recordings", action="store_true")
    ap.add_argument("--keep-io", action="store_true", help="first 1x1 and CTC decoder stay int8")
    ap.add_argument("--no-act-order", action="store_true")
    ap.add_argument("--no-bias-corr", action="store_true")
    ap.add_argument("-j", type=int, default=22)
    a = ap.parse_args()
    E.build()
    img = E.MODEL.read_bytes()
    t0 = time.time()
    Hs = calibrate(img, a.calib, min(a.j, 16))
    print(f"calibrated {len(Hs)} input tensors in {time.time() - t0:.0f}s", flush=True)
    _, ops, _ = graph(img)
    c1 = E.conv1x1_ops(img)
    keep = {c1[0][0], c1[-1][0]} if a.keep_io else set()
    b = bytearray(img)
    err = num = 0.0
    clipped = 0
    hd = struct.unpack_from("<16I", img)
    o_off, blob_off = hd[7], hd[8]
    for i, off, C, N in c1:
        if i in keep:
            continue
        w = np.frombuffer(img, np.int8, C * N, off).reshape(N // 16, C, 16).transpose(0, 2, 1).reshape(N, C)
        H, mx = Hs[ops[i][1]]
        q = gptq(w, H, a.levels, act_order=not a.no_act_order)
        if not a.no_bias_corr:  # keep each output's mean: bias += (W - Q) . E[x], accumulator units
            b_off, = struct.unpack_from("<I", img, o_off + 28 * i + 24)
            if b_off != 0xFFFFFFFF:
                bo = blob_off + b_off
                bias = np.frombuffer(bytes(b[bo: bo + 4 * N]), np.int32).astype(np.int64)
                bias = bias + np.round((w.astype(np.float64) - q) @ mx).astype(np.int64)
                lim = (1 << 19) - 1
                clipped += int((np.abs(bias) > lim).sum())
                b[bo: bo + 4 * N] = np.clip(bias, -lim, lim).astype(np.int32).tobytes()
        err += float(((q.astype(np.float64) - w) ** 2).sum())
        num += float((w.astype(np.float64) ** 2).sum())
        b[off: off + C * N] = q.reshape(N // 16, 16, C).transpose(0, 2, 1).tobytes()
    tag = (f"gptq{a.levels}{'_io' if a.keep_io else ''}{'_noao' if a.no_act_order else ''}"
           f"{'_nobc' if a.no_bias_corr else ''}")
    path = E.OUT / f"citrinet256_cb4_{tag}.mmrt"
    path.write_bytes(bytes(b))
    row = {"variant": tag, "calib": a.calib, "rel_weight_err": round((err / num) ** 0.5, 4), "bias_clipped": clipped}
    sets = {"test-clean": load_split()[:: a.every]}
    if a.recordings:
        from recordings import load_recordings
        sets["recordings"] = load_recordings()
    for name, items in sets.items():
        rows = E.wer(path, items, a.j)
        row[name] = round(score(rows), 3)
        if name == "recordings":
            mine = [r for r in rows if r[0].startswith("me-")]
            row["live_voice"] = round(score(mine), 2) if mine else None
    row["secs"] = round(time.time() - t0)
    print(json.dumps(row), flush=True)
    with open(E.RES / "int4_summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
