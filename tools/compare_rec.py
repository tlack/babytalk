#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""Align a board recording (tools/rec.py --play) to its clean source clip and
report delay, the best-fit gain, and SNR = clean-part power vs everything else
(noise, room echo, mic/codec coloration).

    tools/compare_rec.py data/recordings/x.wav path/to/clean.flac
"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import correlate

rec, rate = sf.read(sys.argv[1], dtype="float32")
clean, crate = sf.read(sys.argv[2], dtype="float32")
assert rate == crate == 16000
for ch in range(rec.shape[1]):
    r = rec[:, ch] - rec[:, ch].mean()
    lag = int(np.argmax(correlate(r, clean, mode="valid", method="fft")))
    seg = r[lag:lag + len(clean)]
    g = float(seg @ clean / (clean @ clean))           # least-squares gain
    resid = seg - g * clean
    snr = 10 * np.log10((g * g * (clean @ clean)) / (resid @ resid))
    noise = r[: max(lag - 1600, 1600)]                  # before playback started
    print(f"mic{ch + 1}: delay {lag / rate * 1000:.0f} ms, gain {g:.3f}, "
          f"SNR vs clean {snr:.1f} dB, speech {20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9):.1f} dBFS, "
          f"noise floor {20 * np.log10(np.sqrt(np.mean(noise ** 2)) + 1e-9):.1f} dBFS")
