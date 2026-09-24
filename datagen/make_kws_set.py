"""Synthetic wake-word test set: positives say the wake phrase (alone or followed by a
command), hard negatives sound similar but don't. 16 kHz wavs + labels.jsonl.

    uv run make_kws_set.py --phrase "hey tomatoface" --out ../data/kws/tomatoface
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tts import Voices

POS = ["{w}", "{w}.", "{w}, turn on the lights", "{w}, what time is it", "okay, {w}", "{w}, play some music"]
NEG = ["hey tomato", "tomato face", "hey there", "tomato sauce", "hey thomas", "the tomatoes are ripe",
       "face the music", "hey tom", "okay computer", "turn on the lights", "hey, it's time to face the day",
       "a tomato fell on my face", "potato face", "hey tamara"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="hey tomatoface")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "kws" / "tomatoface"))
    ap.add_argument("--n-multi", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Voices()
    specs = v.specs(n_multi=a.n_multi)
    rng = np.random.default_rng(1)
    rows = []
    for i, spec in enumerate(specs):
        for speed in (0.9, 1.0, 1.15):
            items = [(t.format(w=a.phrase), 1) for t in rng.choice(POS, 2, replace=False)]
            items += [(t, 0) for t in rng.choice(NEG, 3, replace=False)]
            for text, label in items:
                pcm = v.say(text, spec, speed=speed, noise=float(rng.uniform(0.5, 0.8)))
                pad = np.zeros(int(16000 * rng.uniform(0.2, 0.5)), np.int16)  # leading/trailing silence
                pcm = np.concatenate([pad, pcm, pad])
                name = f"{'pos' if label else 'neg'}-{i:03d}-{speed}-{len(rows):04d}.wav"
                sf.write(out / name, pcm, 16000, subtype="PCM_16")
                rows.append({"wav": name, "text": text, "label": label, "voice": spec[0], "speaker": spec[1],
                             "speed": speed})
    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} clips ({sum(r['label'] for r in rows)} positive) from {len(specs)} voices -> {out}")


if __name__ == "__main__":
    main()
