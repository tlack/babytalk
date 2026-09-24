#!/usr/bin/env python3
"""Wake-phrase demo: the board listens continuously and wakes on a spoken phrase.

    tools/wake.py                                  # "wake up, tomato face" (tools/wake_phrases/)
    tools/wake.py "hello jarvis"                   # any phrase: saved phrase file if there is one,
                                                   # else its plain spelling
    tools/wake.py "hello jarvis" --enroll 3        # say it 3 times first: your voice's spellings
                                                   # are added and saved for next time
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
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RED, DIM, BOLD, GREEN, YEL, RESET = "\033[31m", "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[0m"
PLAIN_THRESHOLD = -15.0  # plain spelling, no TTS evaluation: a middle-of-the-road guess


def slug(phrase):
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def letters(text):
    return "".join(c for c in text.lower() if "a" <= c <= "z")


def enroll(host, phrase, n):
    """Record the phrase n times on the board ('listen'); its transcripts are spellings."""
    heard = []
    for i in range(n):
        input(f"{BOLD}Enroll {i + 1}/{n}:{RESET} press Enter, then say \"{phrase}\" (3 s) ")
        s = socket.create_connection((host, 5555), timeout=60)
        f = s.makefile("rb")
        s.sendall(b"listen 3 1 1 0\n")
        text = ""
        for raw in f:
            line = raw.decode(errors="replace").strip()
            if line.startswith("REC start"):
                print(f"  {RED}● speak now{RESET}")
            elif line.startswith("{") and '"text"' in line:
                text = json.loads(line).get("text", "")
            elif line.startswith("DONE"):
                break
        s.close()
        print(f"  {DIM}heard: {text!r}{RESET}")
        # a transcript far shorter/longer than the phrase is a bad take, not a spelling
        if text and 0.7 <= len(letters(text)) / max(1, len(letters(phrase))) <= 1.5:
            heard.append(text)
        else:
            print(f"  {YEL}skipped (doesn't look like the phrase){RESET}")
    return heard


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="wake phrase (default: the --phrase file)")
    ap.add_argument("--enroll", type=int, default=0, metavar="N", help="say the phrase N times first (your voice)")
    ap.add_argument("--host", default=os.environ.get("STT_HOST", "192.168.1.154"))
    ap.add_argument("--phrase", default=str(ROOT / "wake_phrases" / "wake_up_tomato_face.json"))
    ap.add_argument("--spell", action="append", help="spelling(s) to use instead of a phrase file")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--quiet", action="store_true", help="hide the per-stretch SEG lines")
    a = ap.parse_args()

    if a.spell:
        name, spellings, thr = a.spell[0], a.spell, PLAIN_THRESHOLD
    elif a.text:
        f = ROOT / "wake_phrases" / f"{slug(a.text)}.json"
        if f.exists():
            cfg = json.loads(f.read_text())
            name, spellings, thr = cfg["phrase"], cfg["spellings"], cfg["threshold"]
            print(f"{DIM}using {f.relative_to(ROOT.parent)} ({len(spellings)} spellings){RESET}")
        else:
            name, spellings, thr = a.text, [a.text], PLAIN_THRESHOLD
            if not a.enroll:
                print(f"{YEL}No phrase file for \"{a.text}\": using its plain spelling at threshold {thr}.{RESET}\n"
                      f"{DIM}  For better accuracy: add --enroll 3 (your voice), or build one from TTS voices\n"
                      f"  (export/kws.py --save), and check the phrase first with export/kws_validate.py.{RESET}")
    else:
        cfg = json.loads(Path(a.phrase).read_text())
        name, spellings, thr = cfg["phrase"], cfg["spellings"], cfg["threshold"]
    if a.enroll:
        new = [h for h in enroll(a.host, name, a.enroll) if h not in spellings]
        spellings = spellings + new
        out = ROOT / "wake_phrases" / f"{slug(name)}.json"
        cfg = json.loads(out.read_text()) if out.exists() else {"phrase": name, "threshold": thr}
        cfg["spellings"] = spellings
        out.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{GREEN}enrolled {len(new)} new spelling(s) -> {out.relative_to(ROOT.parent)}{RESET}")
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
