#!/usr/bin/env python3
"""Wake-phrase demo: the board listens continuously and wakes on a spoken phrase.

    tools/wake.py                                  # "wake up, tomato face" (tools/wake_phrases/)
    tools/wake.py --phrase tools/wake_phrases/wake_up_tomato_face.json --threshold -18
    tools/wake.py --spell "hello jarvis" --spell "helo jarvis" --threshold -10

Everything runs on the ESP32-S3: an energy detector gates the speech model, which only
runs on stretches of speech; each one is scored for the wake phrase (CTC keyword
spotting on the STT model's output, stt/main/kws.c). This client just shows the events.
Phrase files come from export/kws.py --save (plain spelling + enrolled spellings).
"""
import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RED, DIM, BOLD, GREEN, YEL, RESET = "\033[31m", "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[0m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("STT_HOST", "192.168.1.154"))
    ap.add_argument("--phrase", default=str(ROOT / "wake_phrases" / "wake_up_tomato_face.json"))
    ap.add_argument("--spell", action="append", help="spelling(s) to use instead of a phrase file")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--quiet", action="store_true", help="hide the per-stretch SEG lines")
    a = ap.parse_args()

    if a.spell:
        name, spellings, thr = a.spell[0], a.spell, -10.0
    else:
        cfg = json.loads(Path(a.phrase).read_text())
        name, spellings, thr = cfg["phrase"], cfg["spellings"], cfg["threshold"]
    if a.threshold is not None:
        thr = a.threshold

    s = socket.create_connection((a.host, 5555), timeout=None)
    f = s.makefile("rb")
    s.sendall(f"wake {thr} {'|'.join(spellings)}\n".encode())
    print(f"{BOLD}Say \"{name}\"{RESET}  {DIM}(threshold {thr}, {len(spellings)} spellings; Ctrl-C to stop){RESET}")
    try:
        for raw in f:
            line = raw.decode(errors="replace").rstrip()
            if line.startswith("LVL "):
                _, floor, peak, speech = line.split()
                bar = "█" * max(0, min(30, int(float(peak) - float(floor))))
                sys.stdout.write(f"\r\033[K{DIM}  listening · room {floor} dBFS · peak {peak} {bar}"
                                 f"{' · speech' if speech == '1' else ''}{RESET}")
                sys.stdout.flush()
                continue
            sys.stdout.write("\r\033[K")
            if line.startswith("SEG "):
                _, ms, score, infer_ms, floor, *text = line.split(" ")
                if not a.quiet:
                    print(f"{DIM}  heard {int(ms) / 1000:.1f}s  score {float(score):7.1f}  "
                          f"({float(infer_ms) / 1000:.2f}s, room {floor} dBFS)  {' '.join(text)!r}{RESET}")
            elif line.startswith("WAKE "):
                print(f"\n{RED}{BOLD}🍅  AWAKE!{RESET}  {DIM}score {line.split()[1]}{RESET}  {YEL}listening for a command…{RESET}")
            elif line.startswith("CMD"):
                text = line[4:].strip()
                print(f"{GREEN}▶ {BOLD}{text or '(no command heard)'}{RESET}\n")
                print(f"{DIM}back to listening for \"{name}\"…{RESET}")
            elif line.startswith("{"):
                info = json.loads(line)
                if "error" in info:
                    sys.exit(f"{RED}error: {info['error']}{RESET}")
            elif line.startswith("DONE"):
                break
    except KeyboardInterrupt:
        s.sendall(b"stop\n")
        print(f"\n{DIM}stopped{RESET}")


if __name__ == "__main__":
    main()
