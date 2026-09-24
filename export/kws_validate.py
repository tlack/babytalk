"""Validate candidate wake phrases before using them: does the model hear them cleanly?

Each phrase is spoken by many TTS voices (datagen/say.py) and transcribed by the
board-identical model. A good wake phrase decodes to itself for nearly every voice, and
its CTC keyword score (plain spelling, no enrollment) sits near 0. Invented words
("tomatoface") don't: the report shows how they get heard, as hints for a better phrase.

  (cd ../datagen && uv run say.py --out ../data/kws/validate "hey tomato face" "hey jarvis")
  uv run kws_validate.py ../data/kws/validate/*
"""
import collections
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from kws import Model, ctc_keyword_score, normalize, phrase_sequences, rel_logprobs, score


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def consistency(heard):
    """Fraction of decodes within 25% letter edit distance of the most central decode."""
    hs = [normalize(h) for h in heard if normalize(h)]
    if not hs:
        return 0.0, ""
    medoid = min(hs, key=lambda h: sum(lev(h, o) for o in hs))
    close = sum(lev(h, medoid) <= 0.25 * max(len(medoid), 1) for h in hs)
    return close / len(heard), medoid


def words(text):
    return [w for w in "".join(c if c.isalpha() or c == " " else " " for c in text.lower()).split()]


def main():
    m = Model()
    rows = []
    for d in map(Path, sys.argv[1:]):
        phrase = d.name.replace("-", " ")
        seqs = phrase_sequences(phrase, m.vocab)
        heard, scores = [], []
        for wav in sorted(d.glob("*.wav")):
            pcm, _ = sf.read(wav, dtype="int16")
            lg = m.logits(pcm)
            from citrinet import ctc_greedy
            import torch
            heard.append(ctc_greedy(torch.from_numpy(lg.T.astype(np.float32)), m.vocab_list))
            scores.append(score(rel_logprobs(lg), seqs)[0])
        target = words(phrase)
        exact = sum(words(h) == target for h in heard)
        per_word = {w: sum(w in words(h) for h in heard) for w in target}
        cons, medoid = consistency(heard)
        rows.append((phrase, exact, len(heard), float(np.median(scores)), per_word,
                     collections.Counter(heard).most_common(4), cons, medoid))
    print(f"{'phrase':22s} {'exact':>6s} {'consistent':>10s} {'score':>6s}  verdict  words heard (of N) / usual decode")
    for phrase, exact, n, med, per_word, common, cons, medoid in sorted(rows, key=lambda r: (-r[6], -r[1], -r[3])):
        # exact: usable with its plain spelling. consistent: usable after enrollment (the
        # model mis-hears it the same way every time, so its own spelling is a template)
        verdict = "GOOD" if exact >= 0.75 * n else ("ENROLL" if cons >= 0.75 else "WEAK")
        pw = " ".join(f"{w}:{k}" for w, k in per_word.items())
        print(f"{phrase:22s} {exact:>3d}/{n:<2d} {100 * cons:>9.0f}% {med:>6.1f}  {verdict:7s}  {pw}")
        if verdict != "GOOD":
            print(f"{'':22s}   usually heard as {medoid!r}; e.g. " + "; ".join(f"{t!r}x{c}" for t, c in common[:3]))


if __name__ == "__main__":
    main()
