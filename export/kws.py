"""CTC keyword spotting on the STT model's output: any wake phrase, no training.

A phrase (or '|'-separated spelling variants) is expanded into every WordPiece
segmentation the model could emit for it; each is scored against the logits with a
"keyword anywhere" CTC Viterbi whose emissions are log-probs relative to each frame's
best token. The score is a log-likelihood ratio: 0 = the phrase is the model's best
reading of those frames, more negative = less like it.

  uv run kws.py --phrase "hey tomatoface|hey tomato face" [--data ../data/kws/tomatoface]
  uv run kws.py --phrase "hey jarvis" --tokens      # just show the spellings
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from citrinet import load_model
from mmrt_check_model import ALLOC, FREE, MODEL, build

BLANK = 256
OUT_EXP = -2  # logits exponent (mmrt model output)


# ----------------------------------------------------------------------------- phrase -> token sequences
# Word boundaries are ignored: a phrase is its letters with spaces removed, split into
# vocab pieces; each piece may be emitted in its word-initial ("fa") or continuation
# ("##fa") form, and scores as the better of the two. So "tomato face" == "tomatoface".
def piece_ids(vocab: dict[str, int]) -> dict[str, tuple[int, ...]]:
    ids: dict[str, list[int]] = {}
    for tok, i in vocab.items():
        if tok.startswith("[") or tok == "'":
            continue
        ids.setdefault(tok[2:] if tok.startswith("##") else tok, []).append(i)
    return {k: tuple(v) for k, v in ids.items()}


def segmentations(letters: str, pieces: dict[str, tuple[int, ...]], cap: int = 64) -> list[tuple[str, ...]]:
    """Splits of `letters` into vocab pieces, longest pieces first, at most `cap`."""
    out: list[tuple[str, ...]] = []

    def rec(pos: int, acc: list[str]):
        if len(out) >= cap:
            return
        if pos == len(letters):
            out.append(tuple(acc))
            return
        for end in range(len(letters), pos, -1):
            if letters[pos:end] in pieces:
                acc.append(letters[pos:end])
                rec(end, acc)
                acc.pop()

    rec(0, [])
    return out


def normalize(text: str) -> str:
    return "".join(c for c in text.lower() if "a" <= c <= "z")


def phrase_sequences(phrase: str, vocab: dict[str, int], cap: int = 64) -> list[tuple[str, ...]]:
    """'|'-separated spellings -> piece sequences (duplicates removed)."""
    pieces = piece_ids(vocab)
    seqs = {}
    for variant in phrase.split("|"):
        for seg in segmentations(normalize(variant), pieces, cap):
            seqs[seg] = True
    return sorted(seqs, key=len)


# ----------------------------------------------------------------------------- scoring
def rel_logprobs(logits_q: np.ndarray) -> np.ndarray:
    """int8 logits [T][257] -> log-softmax minus each frame's max (all <= 0)."""
    z = logits_q.astype(np.float32) * 2.0 ** OUT_EXP
    lp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    return lp - lp.max(axis=1, keepdims=True)


_PIECES: dict[str, tuple[int, ...]] = {}


def ctc_keyword_score(e: np.ndarray, seq: tuple[str, ...]) -> tuple[float, int, int]:
    """Best 'keyword anywhere' CTC Viterbi score of piece sequence `seq` over relative
    log-probs e [T][V]; returns (score, start frame, end frame)."""
    # per-piece emission = best of its token forms; column V is blank
    em = np.stack([e[:, list(_PIECES[p])].max(axis=1) for p in seq] + [e[:, BLANK]], axis=1)
    labels = [len(seq)] + [x for i in range(len(seq)) for x in (i, len(seq))]  # 2L+1 states
    S = len(labels)
    lab = np.array(labels)
    e = em
    skip = np.zeros(S, bool)  # s may come from s-2 (token after a blank, different from s-2)
    for s in range(2, S):
        skip[s] = labels[s] != len(seq) and seq[labels[s]] != seq[labels[s - 2]]
    NEG = -1e9
    alpha = np.full(S, NEG)
    start = np.zeros(S, int)
    best, best_se = NEG, (0, 0)
    for t in range(e.shape[0]):
        stay = alpha
        prev1 = np.concatenate([[NEG], alpha[:-1]])
        prev2 = np.where(skip, np.concatenate([[NEG, NEG], alpha[:-2]]), NEG)
        st1 = np.concatenate([[0], start[:-1]])
        st2 = np.concatenate([[0, 0], start[:-2]])
        cand = np.stack([stay, prev1, prev2])
        arg = cand.argmax(axis=0)
        new = cand.max(axis=0)
        new_start = np.choose(arg, [start, st1, st2])
        for s0 in (0, 1):  # free start at this frame
            if new[s0] < 0:
                new[s0], new_start[s0] = 0.0, t
        alpha = new + e[t, lab]
        start = new_start
        for s_end in (S - 2, S - 1):  # after the last token (or its trailing blank)
            if alpha[s_end] > best:
                best, best_se = alpha[s_end], (start[s_end], t)
    return float(best), int(best_se[0]), int(best_se[1])


def score(e: np.ndarray, seqs: list[tuple[str, ...]]) -> tuple[float, tuple[str, ...]]:
    best = (-1e9, None)
    for seq in seqs:
        s, _, _ = ctc_keyword_score(e, seq)
        if s > best[0]:
            best = (s, seq)
    return best


# ----------------------------------------------------------------------------- logits via the host mmrt runtime
class Model:
    """Board-identical int8 logits: NeMo-exact features -> int8 -> mmrt (host build)."""

    def __init__(self):
        self.lib = build()
        libc = ctypes.CDLL(None)
        libc.malloc.restype = ctypes.c_void_p
        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.free.argtypes = [ctypes.c_void_p]
        self.alloc, self.free = ALLOC(lambda n: libc.malloc(n)), FREE(lambda p: libc.free(p))
        img = MODEL.read_bytes()
        self.buf = ctypes.create_string_buffer(img, len(img))
        self.model = ctypes.create_string_buffer(64)
        assert self.lib.mmrt_open(self.model, self.buf, len(img)) == 0
        self.lib.mmrt_run.restype = ctypes.POINTER(ctypes.c_int8)
        self.lib.mmrt_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int), ALLOC, FREE]
        _, self.feat, self.vocab_list, _ = load_model()
        self.vocab = {t: i for i, t in enumerate(self.vocab_list)}
        _PIECES.update(piece_ids(self.vocab))

    def transcribe(self, pcm_int16: np.ndarray) -> str:
        from citrinet import ctc_greedy
        return ctc_greedy(torch.from_numpy(self.logits(pcm_int16).T.astype(np.float32)), self.vocab_list)

    def logits_cached(self, key: str, pcm_int16: np.ndarray) -> np.ndarray:
        cache = Path(__file__).resolve().parent.parent / "data" / "kws" / "logits_cache"
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / (key.replace("/", "_") + ".npy")
        if f.exists():
            return np.load(f)
        lg = self.logits(pcm_int16)
        np.save(f, lg)
        return lg

    def logits(self, pcm_int16: np.ndarray) -> np.ndarray:
        x = self.feat(torch.from_numpy(pcm_int16.astype(np.float32) / 32768.0)).numpy().T  # [T, 80]
        q = np.clip(np.floor(x * 32 + 0.5), -128, 127).astype(np.int8)  # input exponent -5
        q = np.ascontiguousarray(q)
        T_out = ctypes.c_int()
        out = self.lib.mmrt_run(self.model, q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), q.shape[0],
                                ctypes.byref(T_out), self.alloc, self.free)
        return np.ctypeslib.as_array(out, shape=(T_out.value * 272,)).reshape(T_out.value, 272)[:, :257].copy()


def ctc_text(lg: np.ndarray, vocab_list) -> str:
    from citrinet import ctc_greedy
    return ctc_greedy(torch.from_numpy(lg.T.astype(np.float32)), vocab_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="hey tomatoface|hey tomato face")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "kws" / "tomatoface"))
    ap.add_argument("--libri", type=int, default=200, help="also score N LibriSpeech test-clean utterances (negatives)")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--enroll", type=int, default=8, help="voices whose positives enroll spellings (held out of eval)")
    a = ap.parse_args()
    m = Model()
    rows = [json.loads(l) for l in open(Path(a.data) / "labels.jsonl")]
    voices = sorted({(r["voice"], r["speaker"]) for r in rows}, key=str)
    enroll_voices = set(voices[:: max(1, len(voices) // a.enroll)][: a.enroll]) if a.enroll else set()
    phrase = a.phrase
    if enroll_voices:
        heard = []
        for r in rows:
            if r["label"] and (r["voice"], r["speaker"]) in enroll_voices and r["text"].rstrip(".").lower() == a.phrase.split("|")[0]:
                pcm, _ = sf.read(Path(a.data) / r["wav"], dtype="int16")
                lg = m.logits_cached(Path(a.data).name + "-" + r["wav"], pcm)
                heard.append(ctc_text(lg, m.vocab_list))
        # keep spellings of plausible length: a template much shorter than the phrase
        # ("hate my face") would match ordinary speech
        n = len(normalize(a.phrase.split("|")[0]))
        heard = [h for h in heard if 0.75 * n <= len(normalize(h)) <= 1.4 * n]
        print(f"enrolled from {len(enroll_voices)} voices: " + "; ".join(repr(h) for h in heard))
        phrase = "|".join([a.phrase] + heard)
        rows = [r for r in rows if (r["voice"], r["speaker"]) not in enroll_voices]
    seqs = phrase_sequences(phrase, m.vocab)
    print(f"{len(seqs)} piece sequences, e.g. " + "; ".join(" ".join(s) for s in seqs[:4]))
    if a.tokens:
        return
    res = []
    for r in rows:
        pcm, _ = sf.read(Path(a.data) / r["wav"], dtype="int16")
        s, seq = score(rel_logprobs(m.logits_cached(Path(a.data).name + "-" + r["wav"], pcm)), seqs)
        res.append((s, r["label"], r["text"], r["voice"]))
    if a.libri:
        from citrinet import load_split
        libri = load_split("test-clean") + load_split("dev-clean")
        hours = 0.0
        for uid, path, text in libri[:: max(1, len(libri) // a.libri)][: a.libri]:
            pcm, _ = sf.read(path, dtype="int16")
            hours += len(pcm) / 16000 / 3600
            s, _ = score(rel_logprobs(m.logits_cached("libri-" + uid, pcm)), seqs)
            res.append((s, 0, "libri:" + text[:40], "librispeech"))
    if a.libri:
        print(f"LibriSpeech negatives: {a.libri} utterances, {hours:.2f} h of speech")
    pos = np.array([s for s, l, *_ in res if l])
    neg = np.array([s for s, l, *_ in res if not l])
    print(f"positives: {len(pos)}  median {np.median(pos):.1f}  10th pct {np.percentile(pos, 10):.1f}  min {pos.min():.1f}")
    print(f"negatives: {len(neg)}  median {np.median(neg):.1f}  90th pct {np.percentile(neg, 90):.1f}  max {neg.max():.1f}")
    for thr in sorted({round(x, 1) for x in [-2, -4, -6, -8, -10, -12, -15]}, reverse=True):
        print(f"  threshold {thr:6.1f}: detect {100 * (pos >= thr).mean():5.1f}%   false wakes {int((neg >= thr).sum())}/{len(neg)}")
    print("hardest negatives:")
    for s, l, text, voice in sorted([r for r in res if not r[1]], reverse=True)[:6]:
        print(f"  {s:7.1f}  {text!r}  ({voice})")
    print("weakest positives:")
    for s, l, text, voice in sorted([r for r in res if r[1]])[:6]:
        print(f"  {s:7.1f}  {text!r}  ({voice})")


if __name__ == "__main__":
    main()
