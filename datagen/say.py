"""Speak one or more phrases with many TTS voices -> 16 kHz wavs (for wake-phrase validation).

    uv run say.py --out /tmp/v "hey tomato face" "hey computer" [--voices 12]
Writes <out>/<phrase-slug>/<i>.wav and prints the directory per phrase.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from tts import Voices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phrases", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", type=int, default=12)
    a = ap.parse_args()
    v = Voices()
    # all single-speaker voices + multi-speaker ones, trimmed to --voices (fixed seed)
    specs = v.specs(n_multi=a.voices)[: a.voices]
    for phrase in a.phrases:
        d = Path(a.out) / re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")
        d.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(specs):
            pcm = v.say(phrase, spec, speed=1.0)
            pad = np.zeros(4000, np.int16)
            sf.write(d / f"{i:02d}.wav", np.concatenate([pad, pcm, pad]), 16000, subtype="PCM_16")
        print(d)


if __name__ == "__main__":
    main()
