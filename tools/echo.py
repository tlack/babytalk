#!/usr/bin/env python3
"""Talk to the board, hear it read back: speech to text AND text to speech, both on the
ESP32-S3 (MicroPython firmware with the stt + tts modules, see mpy/README.md).

    uv run --with mpremote tools/echo.py              # Enter -> speak for 4 s -> it replies
    uv run --with mpremote tools/echo.py --secs 6     # longer takes
    uv run --with mpremote tools/echo.py --prefix ""  # just repeat, no "You said:"

The PC only sends "take" and prints what the board reports; recording, transcription
and synthesis all run on the board. Board files (echo_board.py + the codec drivers in
mpy/drivers/) are pushed when missing or changed.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mpremote.transport_serial import SerialTransport

ROOT = Path(__file__).resolve().parent.parent
DRIVERS = ROOT / "mpy/drivers"
FILES = {
    "echo_board.py": ROOT / "mpy/examples/echo_board.py",
    "io_ext.py": DRIVERS / "io_ext.py",
    "es7210.py": DRIVERS / "es7210.py",
    "es8311.py": DRIVERS / "es8311.py",
}
RED, GREEN, DIM, BOLD, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def push(t):
    for dest, src in FILES.items():
        data = src.read_bytes()
        try:
            same = t.fs_hashfile(dest, "sha256") == hashlib.sha256(data).digest()
        except Exception:
            same = False
        if not same:
            print(f"{DIM}push {dest}{RESET}")
            t.fs_writefile(dest, data)


PLAY = {"win": None}


def stage_wav(path):
    """Copy a PCM WAV where Windows' SoundPlayer can reach it (WSL): --play testing aid."""
    out = subprocess.check_output(["cmd.exe", "/c", "echo %TEMP%"], text=True, stderr=subprocess.DEVNULL).strip()
    tmp = Path(subprocess.check_output(["wslpath", "-u", out], text=True).strip()) / "mm_echo_play.wav"
    shutil.copy(path, tmp)
    return subprocess.check_output(["wslpath", "-w", str(tmp)], text=True).strip()


def show(line):
    line = line.strip("\x04\r\n >")
    if line.startswith("REC"):
        if PLAY["win"]:
            subprocess.Popen(["powershell.exe", "-NoProfile", "-Command",
                              f"(New-Object Media.SoundPlayer '{PLAY['win']}').PlaySync()"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{RED}{BOLD}● speak now{RESET} {DIM}({line.split()[1]} s){RESET}", flush=True)
    elif line.startswith("PROCESSING"):
        print(f"{DIM}  thinking...{RESET}", flush=True)
    elif line.startswith("HEARD"):
        text = line[6:].strip()
        print(f"  heard: {BOLD}{text or '(nothing)'}{RESET}", flush=True)
    elif line.startswith(("STT", "TTS")):
        print(f"{DIM}  {line}{RESET}", flush=True)
    elif line.startswith("DONE"):
        print(f"{GREEN}  ▶ spoken{RESET}\n", flush=True)
    elif line and not line.startswith("READY"):
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--secs", type=float, default=4.0, help="recording length per take")
    ap.add_argument("--prefix", default="You said: ", help='spoken before the transcript ("" to just repeat)')
    ap.add_argument("--takes", type=int, default=0, help="stop after N takes (0: until Ctrl-C)")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for Enter between takes")
    ap.add_argument("--play", help="testing: a WAV to play through the laptop speaker as each take starts")
    a = ap.parse_args()
    if a.play:
        PLAY["win"] = stage_wav(a.play)

    t = SerialTransport(a.port, baudrate=115200)
    buf = [""]

    def consume(data):
        buf[0] += data.decode(errors="replace")
        while "\n" in buf[0]:
            line, buf[0] = buf[0].split("\n", 1)
            show(line)

    try:
        t.enter_raw_repl()
        push(t)
        t.exec("import echo_board; echo_board.setup()", data_consumer=consume)
        n = 0
        while not a.takes or n < a.takes:
            if not a.no_wait:
                input(f"{BOLD}Press Enter, then talk ({a.secs:g} s){RESET} ")
            t.exec(f"echo_board.take({a.secs}, {a.prefix!r})", data_consumer=consume)
            n += 1
    except (KeyboardInterrupt, EOFError):
        print(f"\n{DIM}bye{RESET}")
    finally:
        try:
            t.exit_raw_repl()
        except Exception:
            pass
        t.close()


if __name__ == "__main__":
    sys.exit(main())
