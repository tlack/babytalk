"""Piper TTS -> 16 kHz int16 mono, across many voices/speakers (data/tts_voices).

    from tts import Voices
    v = Voices()
    for spec in v.specs(n_multi=20):          # (voice, speaker_id) pairs
        pcm = v.say("hey tomatoface", spec, speed=1.1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soxr
from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "data" / "tts_voices"


class Voices:
    def __init__(self, voice_dir: Path = VOICES):
        self.paths = sorted(voice_dir.glob("*.onnx"))
        self._cache: dict[str, PiperVoice] = {}

    def _get(self, name: str) -> PiperVoice:
        if name not in self._cache:
            self._cache[name] = PiperVoice.load(VOICES / f"{name}.onnx")
        return self._cache[name]

    def specs(self, n_multi: int = 20, seed: int = 0):
        """Every single-speaker voice, plus n_multi speakers of each multi-speaker voice."""
        rng = np.random.default_rng(seed)
        out = []
        for p in self.paths:
            v = self._get(p.stem)
            n = v.config.num_speakers
            if n <= 1:
                out.append((p.stem, None))
            else:
                out += [(p.stem, int(s)) for s in rng.choice(n, size=min(n_multi, n), replace=False)]
        return out

    def say(self, text: str, spec, speed: float = 1.0, noise: float = 0.667) -> np.ndarray:
        name, speaker = spec
        v = self._get(name)
        cfg = SynthesisConfig(speaker_id=speaker, length_scale=1.0 / speed, noise_scale=noise)
        chunks = [c.audio_float_array for c in v.synthesize(text, cfg)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        audio = soxr.resample(audio, v.config.sample_rate, 16000)
        return np.clip(audio * 32767, -32768, 32767).astype(np.int16)
