"""WER evaluation of the Citrinet port on LibriSpeech.

  uv run evalwer.py float            # our port, exact lengths (reference; NeMo-equivalent)
  uv run evalwer.py float --fold     # same with BatchNorm folded
  uv run evalwer.py fixed            # static 1600-frame window, pad / chunk, no masking
  uv run evalwer.py onnx             # the exported static ONNX via onnxruntime (fixed-window semantics)

Per-utterance hypotheses -> data/results/<tag>.tsv, summary printed + appended to
data/results/summary.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jiwer
import torch

from citrinet import DATA, ctc_greedy, fold_bn, load_model, load_split, normalize_text, read_audio

WIN = 1600  # static input: 16 s of 10 ms feature frames
SUB = 8  # encoder time downsampling
RESULTS = DATA / "results"


def fixed_windows(feats: torch.Tensor, win: int = WIN):
    """feats: un-normalized log-mel [80, T]. Returns list of (normalized padded [80,win], n_valid).
    Short clips: normalize over valid frames, zero-pad to win (NeMo's pad value).
    Long clips: split into ceil(T/win) equal chunks, each normalized on its own."""
    from citrinet import MelFeatures

    T = feats.shape[1]
    n = max(1, math.ceil(T / win))
    step = math.ceil(T / n)
    out = []
    for i in range(n):
        c = feats[:, i * step : min(T, (i + 1) * step)]
        c = MelFeatures.normalize(c)[0]
        v = c.shape[1]
        out.append((torch.nn.functional.pad(c, (0, win - v)), v))
    return out


def out_len(n_in: int) -> int:
    n = n_in
    for _ in range(3):  # three stride-2 convs, k odd, pad (k-1)/2 -> floor((n-1)/2)+1
        n = (n - 1) // 2 + 1
    return n


_G = {}


def _init(mode, fold, pad_mode, onnx_path):
    torch.set_num_threads(1)
    m, feat, vocab, cfg = load_model()
    feat.pad_mode = pad_mode
    if fold:
        m = fold_bn(m)
    _G.update(m=m, feat=feat, vocab=vocab, mode=mode)
    if mode == "onnx":
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _G["sess"] = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])


def _run_static(x: torch.Tensor) -> torch.Tensor:
    """x [1,80,WIN] -> logits [257, WIN/8]."""
    if _G["mode"] == "onnx":
        s = _G["sess"]
        return torch.from_numpy(s.run(None, {s.get_inputs()[0].name: x.numpy()})[0])[0]
    return _G["m"](x)[0][0]


@torch.no_grad()
def _one(item):
    uid, path, ref = item
    audio = read_audio(path)
    feat, m, vocab, mode = _G["feat"], _G["m"], _G["vocab"], _G["mode"]
    if mode == "float":
        x = feat(audio)[None]
        lg, _ = m(x)
        hyp = ctc_greedy(lg[0], vocab)
    else:
        lm = feat.logmel(audio)[:, : feat.seq_len(audio.shape[0])]
        parts = []
        for x, v in fixed_windows(lm):
            lg = _run_static(x[None])
            parts.append(ctc_greedy(lg, vocab, out_len(v)))
        hyp = " ".join(p for p in parts if p)
    return uid, ref, hyp, audio.shape[0] / 16000


def score(rows):
    refs = [normalize_text(r[1]) for r in rows]
    hyps = [normalize_text(r[2]) for r in rows]
    return 100 * jiwer.wer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["float", "fixed", "onnx"])
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--fold", action="store_true")
    ap.add_argument("--pad-mode", default="constant", help="STFT centre padding: constant (NeMo now) / reflect")
    ap.add_argument("--onnx", default=str(DATA / "models" / "citrinet256_static1600.onnx"))
    ap.add_argument("-n", type=int, default=0, help="limit utterances (0 = all)")
    ap.add_argument("-j", type=int, default=12)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    items = load_split(a.split)
    if a.n:
        items = items[: a.n]
    tag = a.tag or f"{a.mode}{'_fold' if a.fold else ''}_{a.pad_mode}_{a.split}_{len(items)}"
    t0 = time.time()
    with ProcessPoolExecutor(a.j, initializer=_init, initargs=(a.mode, a.fold, a.pad_mode, a.onnx)) as ex:
        rows = list(ex.map(_one, items, chunksize=8))
    wer = score(rows)
    short = [r for r in rows if r[3] <= WIN / 100]
    long_ = [r for r in rows if r[3] > WIN / 100]
    summ = dict(tag=tag, mode=a.mode, fold=a.fold, pad_mode=a.pad_mode, split=a.split, n=len(rows),
                wer=round(wer, 3), n_le16s=len(short), wer_le16s=round(score(short), 3) if short else None,
                n_gt16s=len(long_), wer_gt16s=round(score(long_), 3) if long_ else None,
                secs=round(time.time() - t0, 1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{tag}.tsv", "w") as f:
        for uid, ref, hyp, dur in rows:
            f.write(f"{uid}\t{dur:.2f}\t{ref}\t{hyp}\n")
    with open(RESULTS / "summary.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
