"""Check the board's front end (stt firmware `fe` command) against the Python features:
PCM goes to the board, its int8 model input comes back and is compared with the
Python pipeline's (round half up at exponent -5). Also reports the board's time.

  uv run check_frontend_device.py [--host IP]
"""
import argparse
import json
import os
import socket

import numpy as np
import soundfile as sf
import torch

from citrinet import load_model, load_split
from recordings import load_recordings


def board_fe(host, pcm):
    s = socket.create_connection((host, 5555), timeout=120)
    f = s.makefile("rb")
    s.sendall(f"fe {len(pcm)}\n".encode() + pcm.astype("<i2").tobytes())
    info = {}
    while True:
        line = f.readline().decode().strip()
        if line.startswith("FEATS "):
            q = np.frombuffer(f.read(int(line.split()[1])), dtype=np.int8)
        elif line.startswith("{"):
            info.update(json.loads(line))
        elif line.startswith("DONE"):
            return q.reshape(info["T"], 80), info["fe_us"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("STT_HOST", "192.168.1.154"))
    a = ap.parse_args()
    _, feat, _, _ = load_model()
    items = [p for _, p, _ in load_recordings()] + [p for _, p, _ in load_split()[::262]]
    bad = total = 0
    worst = 0
    secs = fe_s = 0.0
    for path in items:
        x, _ = sf.read(str(path), dtype="int16")
        x = x[:, 0].copy() if x.ndim == 2 else x
        q_py = np.clip(np.floor(feat(torch.from_numpy(x.astype(np.float32) / 32768)).numpy().T * 32 + 0.5), -128, 127)
        q_b, us = board_fe(a.host, x)
        d = np.abs(q_py.astype(int) - q_b.astype(int))
        bad += int((d > 0).sum())
        total += d.size
        worst = max(worst, int(d.max()))
        secs += len(x) / 16000
        fe_s += us / 1e6
    print(f"{len(items)} clips: {bad}/{total} int8 values differ ({100 * bad / total:.3f}%), worst {worst} LSB; "
          f"board front end {1000 * fe_s / secs:.1f} ms per second of audio")


if __name__ == "__main__":
    main()
