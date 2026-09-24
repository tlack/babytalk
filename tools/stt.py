#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial>=3.5", "numpy", "soundfile"]
# ///
"""Drive the on-device STT test firmware (stt/) over WiFi (TCP port 5555).

    tools/stt.py pcm data/recordings/me-*.wav [--mode 1|2] [--host 192.168.1.154]
    tools/stt.py load [internal_kb] [param_copy]

The board's address comes from --host, $STT_HOST, or by asking its USB console
(`ip`). pcm sends 16 kHz mono s16 PCM (mic1 of a stereo board recording) and prints
the on-device transcript + timing; results append to results/<date>-stt.jsonl.
"""
import argparse
import datetime
import json
import os
import pathlib
import socket
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench import git_rev  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = 5555


def board_ip(port="/dev/ttyACM0"):
    from bench import open_port, wait_ready
    ser = open_port(port)
    wait_ready(ser, prompt=b"stt>")
    for _ in range(30):
        ser.write(b"ip\n")
        time.sleep(1)
        for line in ser.read(4096).decode(errors="replace").splitlines():
            if line.startswith("@@IP") and "0.0.0.0" not in line:
                return line.split()[1].split(":")[0]
    sys.exit("board has no IP (WiFi not up?)")


class Board:
    def __init__(self, host):
        self.s = socket.create_connection((host, PORT), timeout=300)
        self.f = self.s.makefile("rb")

    def request(self, line, payload=b""):
        """-> (list of JSON replies, raw logits bytes or None, rc)."""
        self.s.sendall(line.encode() + b"\n" + payload)
        replies, logits = [], None
        while True:
            raw = self.f.readline()
            if not raw:
                sys.exit("connection closed")
            text = raw.decode(errors="replace").rstrip()
            if text.startswith("DONE "):
                return replies, logits, int(text.rsplit("rc=", 1)[1])
            if text.startswith("LOGITS "):
                logits = self.f.read(int(text.split()[1]))
            elif text.startswith("{"):
                replies.append(json.loads(text))
            elif text:
                print(text)


def mono16k(path):
    a, rate = sf.read(path, dtype="int16")
    assert rate == 16000, f"{path}: {rate} Hz"
    return a[:, 0].copy() if a.ndim == 2 else a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("STT_HOST"))
    ap.add_argument("--serial", default="/dev/ttyACM0")
    ap.add_argument("--mode", type=int, default=1, help="ESP-DL runtime: 0 auto, 1 single core, 2 multi core")
    ap.add_argument("--static", action="store_true", help="use the fixed 1600-frame window instead of an exact-length graph")
    ap.add_argument("--note", default="")
    ap.add_argument("cmd", choices=["load", "pcm"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()

    host = a.host or board_ip(a.serial)
    b = Board(host)
    if a.cmd == "load":
        for r in b.request(" ".join(["load"] + a.args))[0]:
            print(json.dumps(r))
        return
    out = ROOT / "results" / f"{datetime.date.today()}-stt.jsonl"
    meta = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "git": git_rev(),
            "note": a.note, "host": host}
    with out.open("a") as f:
        for path in a.args:
            pcm = mono16k(path)
            replies, _, rc = b.request(f"pcm {len(pcm)} {a.mode} {0 if a.static else 1}", pcm.astype("<i2").tobytes())
            info = {k: v for r in replies for k, v in r.items()}
            print(f"{pathlib.Path(path).name:32s} {info.get('total_ms', 0):7.0f} ms "
                  f"(audio {info.get('audio_ms', 0):5.0f}, fe {info.get('fe_ms', 0):5.0f}, "
                  f"model {info.get('model_ms', 0):6.0f}, load {info.get('load_ms', 0):4.0f})  "
                  f"RTF {info.get('rtf', 0):.2f} rc={rc} -> {info.get('text')!r}")
            f.write(json.dumps({**meta, "clip": str(path), "mode": a.mode, "static": a.static, "rc": rc, **info}) + "\n")


if __name__ == "__main__":
    main()
