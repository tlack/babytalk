#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial>=3.5", "numpy", "soundfile"]
# ///
"""Talk to the board, see what it heard. Recording, speech-to-text and silence
trimming all happen on the ESP32-S3; this just shows it.

    tools/listen.py            # 4 s recording
    tools/listen.py 6          # 6 s
    tools/listen.py 3 --loop   # Enter to record again, Ctrl-C to quit
    tools/listen.py --save     # also keep the audio in data/recordings/live-*.wav

The board has no LED, so the terminal is the indicator: a live mic-level meter and
countdown while it records, a timer while it thinks, then the transcript.
"""
import argparse
import datetime
import json
import os
import pathlib
import select
import socket
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stt import PORT, board_ip  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RED, DIM, BOLD, GREEN, RESET = "\033[31m", "\033[2m", "\033[1m", "\033[32m", "\033[0m"
CLEAR = "\r\033[K"


class Conn:
    def __init__(self, host):
        self.s = socket.create_connection((host, PORT), timeout=10)
        self.buf = b""

    def send(self, line):
        self.s.sendall(line.encode() + b"\n")

    def line(self, timeout):
        """Next line, or None if nothing arrives within timeout."""
        while b"\n" not in self.buf:
            r, _, _ = select.select([self.s], [], [], timeout)
            if not r:
                return None
            chunk = self.s.recv(65536)
            if not chunk:
                sys.exit("board closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="replace")

    def exact(self, n):
        while len(self.buf) < n:
            self.buf += self.s.recv(65536)
        data, self.buf = self.buf[:n], self.buf[n:]
        return data


def meter(db, width=30):
    frac = min(1.0, max(0.0, (db + 70) / 60))  # -70 dBFS .. -10 dBFS
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def listen_once(c, secs, args):
    c.send(f"listen {secs} {args.mode} {0 if args.no_trim else 1} {1 if args.save else 0}")
    info, audio, t_done = {}, None, None
    spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while True:
        line = c.line(0.1)
        if line is None:
            if t_done:  # thinking: animate
                i += 1
                sys.stdout.write(f"{CLEAR}{spin[i % len(spin)]} thinking… {time.monotonic() - t_done:4.1f}s")
                sys.stdout.flush()
            continue
        if line.startswith("REC start"):
            sys.stdout.write(f"{CLEAR}{RED}●{RESET} speak now…")
            sys.stdout.flush()
        elif line.startswith("LEVEL "):
            _, db, ms = line.split()
            left = secs - int(ms) / 1000
            sys.stdout.write(f"{CLEAR}{RED}● REC{RESET} {left:4.1f}s left  ▕{meter(float(db))}▏ {float(db):6.1f} dBFS")
            sys.stdout.flush()
        elif line.startswith("REC done"):
            t_done = time.monotonic()
        elif line.startswith("AUDIO "):
            audio = c.exact(int(line.split()[1]))
        elif line.startswith("{"):
            info.update(json.loads(line))
        elif line.startswith("DONE "):
            break
        elif line.strip():
            print(f"{CLEAR}{DIM}{line}{RESET}")
    sys.stdout.write(CLEAR)
    if "error" in info:
        print(f"{RED}error: {info['error']}{RESET}")
        return info
    text = info.get("text", "")
    print(f"{GREEN}▶{RESET} {BOLD}{text or '(nothing heard)'}{RESET}")
    print(f"{DIM}  speech {info.get('speech_ms', 0) / 1000:.1f}s of {info.get('recorded_ms', 0) / 1000:.0f}s · "
          f"waited {info.get('wait_ms', 0) / 1000:.1f}s (features {info.get('fe_ms', 0) / 1000:.1f}s, "
          f"model {info.get('model_ms', 0) / 1000:.1f}s"
          f"{', graph rebuild %.1fs' % (info['load_ms'] / 1000) if 'load_ms' in info else ''}){RESET}")
    if audio is not None:
        save(audio, text, info)
    return info


def save(audio, text, info):
    import numpy as np
    import soundfile as sf
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "data" / "recordings" / f"live-{ts}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.frombuffer(audio, dtype="<i2"), 16000, subtype="PCM_16")
    meta = {"secs": info.get("recorded_ms", 0) / 1000, "gain": 14, "rate": 16000, "channels": ["mic1"],
            "played": None, "text": "", "hyp_on_device": text, "recorded": ts,
            "note": "tools/listen.py --save; fill in `text` with what was actually said to use as a test clip"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{DIM}  saved {out.relative_to(ROOT)}{RESET}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("secs", nargs="?", type=int, default=4, help="recording length, 1-15 s")
    ap.add_argument("--host", default=os.environ.get("STT_HOST"))
    ap.add_argument("--serial", default="/dev/ttyACM0")
    ap.add_argument("--loop", action="store_true", help="Enter to record again")
    ap.add_argument("--save", action="store_true", help="keep the audio in data/recordings/")
    ap.add_argument("--no-trim", action="store_true", help="transcribe the whole recording")
    ap.add_argument("--mode", type=int, default=2, help="ESP-DL runtime: 0 auto, 1 single, 2 multi core")
    args = ap.parse_args()

    host = args.host or board_ip(args.serial)
    c = Conn(host)
    c.s.settimeout(None)
    try:
        while True:
            listen_once(c, args.secs, args)
            if not args.loop:
                break
            input(f"{DIM}Enter to record again, Ctrl-C to quit{RESET} ")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
