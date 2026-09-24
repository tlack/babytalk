"""Check the C wake-phrase scorer (stt/main/kws.c) against the Python one (kws.py) on the
same int8 logits. Scores must agree to float rounding.

  uv run kws_check_c.py [--phrase ...] [--data ../data/kws/wakeup]
"""
import argparse
import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from kws import Model, phrase_sequences, rel_logprobs, score

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "mmrt" / "build" / "libkws.so"


class Phrase(ctypes.Structure):
    _fields_ = [("n_seqs", ctypes.c_int), ("len", ctypes.c_uint8 * 512), ("piece", (ctypes.c_uint16 * 40) * 512)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="wake up tomato face")
    ap.add_argument("--data", default=str(ROOT / "data" / "kws" / "wakeup"))
    ap.add_argument("-n", type=int, default=40)
    a = ap.parse_args()
    LIB.parent.mkdir(exist_ok=True)
    subprocess.check_call(["gcc", "-O2", "-Wall", "-shared", "-fPIC", "-o", str(LIB), str(ROOT / "stt/main/kws.c"), "-lm"])
    lib = ctypes.CDLL(str(LIB))
    m = Model()
    vocab = (ctypes.c_char_p * len(m.vocab_list))(*[t.encode() for t in m.vocab_list])
    lib.kws_init(vocab, len(m.vocab_list))
    ph = Phrase()
    for sp in a.phrase.split("|"):
        lib.kws_add_spelling(ctypes.byref(ph), sp.encode(), 64)
    lib.kws_score.restype = ctypes.c_float
    lib.kws_score.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    seqs = phrase_sequences(a.phrase, m.vocab)
    print(f"sequences: C {ph.n_seqs}, Python {len(seqs)}")
    rows = [json.loads(l) for l in open(Path(a.data) / "labels.jsonl")][: a.n]
    worst = 0.0
    for r in rows:
        pcm, _ = sf.read(Path(a.data) / r["wav"], dtype="int16")
        lg = np.ascontiguousarray(m.logits_cached(Path(a.data).name + "-" + r["wav"], pcm))
        py, _ = score(rel_logprobs(lg), seqs)
        # kws.py's rel_logprobs is log-softmax minus its max == (z - max z): same quantity
        st, en = ctypes.c_int(), ctypes.c_int()
        c = lib.kws_score(ctypes.byref(ph), lg.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), lg.shape[0], lg.shape[1],
                          -2, ctypes.byref(st), ctypes.byref(en))
        worst = max(worst, abs(c - py))
    print(f"{len(rows)} clips: max |C - Python| = {worst:.4f}")


if __name__ == "__main__":
    main()
