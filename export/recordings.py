"""Board mic recordings (tools/rec.py) as an eval set for the Citrinet pipeline.

Each data/recordings/*.json with a transcript becomes an item (uid, wav, ref):
`ref` if present (spoken-form normalization, e.g. digits), else `text`. A replayed
LibriSpeech clip gets its transcript from LibriSpeech, and its clean source is
added too as "clean:<uid>", so board-mic vs clean WER compare the same sentence.

  uv run recordings.py           # float, exact lengths, per-clip hypotheses
"""
from __future__ import annotations

import json
from pathlib import Path

from citrinet import DATA, LIBRI

REC = DATA / "recordings"


def libri_transcript(flac: Path) -> str:
    uid = flac.stem
    spk, chap, _ = uid.split("-")
    for line in open(flac.parent / f"{spk}-{chap}.trans.txt"):
        if line.startswith(uid + " "):
            return line.split(" ", 1)[1].strip().lower()
    raise KeyError(uid)


def load_recordings():
    items = []
    for j in sorted(REC.glob("*.json")):
        m = json.loads(j.read_text())
        wav = j.with_suffix(".wav")
        if m.get("played"):
            src = Path(m["played"])
            src = src if src.is_absolute() else DATA.parent / src
            ref = libri_transcript(src)
            items.append((f"clean:{src.stem}", src, ref))
        else:
            ref = m.get("ref") or m.get("text")
        if ref:
            items.append((j.stem, wav, ref))
    return items


if __name__ == "__main__":
    import torch

    import evalwer
    from citrinet import normalize_text

    evalwer._init("float", fold=False, pad_mode="constant", onnx_path=None)
    rows = [evalwer._one(it) for it in load_recordings()]
    for uid, ref, hyp, dur in rows:
        import jiwer
        w = 100 * jiwer.wer(normalize_text(ref), normalize_text(hyp))
        print(f"{uid:34s} {w:5.1f}%  ref: {normalize_text(ref)}\n{'':41s} hyp: {normalize_text(hyp)}")
    mine = [r for r in rows if r[0].startswith("me-")]
    print(f"\nlive voice ({len(mine)} clips): WER {evalwer.score(mine):.2f}%")
