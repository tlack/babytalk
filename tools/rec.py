#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial>=3.5", "numpy", "soundfile"]
# ///
"""Record the board's mics (bench firmware `rec` command) to a WAV on the laptop.

    tools/rec.py --secs 8 --out data/recordings/me-hello.wav
    tools/rec.py --play data/librispeech/.../1089-134686-0000.flac --out data/recordings/ls-1089-0000.wav

With --play, the clip is played through the laptop's speaker (Windows audio, from
WSL) as soon as the board reports that capture has started, and the recording
length defaults to the clip length + 2 s. Writes the stereo capture (mic1=L,
mic2=R) plus a JSON sidecar with gain, levels and the source clip.
"""
import argparse
import base64
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench import open_port, wait_ready  # noqa: E402

RATE = 16000


def windows_temp():
    out = subprocess.check_output(["cmd.exe", "/c", "echo %TEMP%"], text=True,
                                  stderr=subprocess.DEVNULL).strip()
    return pathlib.Path(subprocess.check_output(["wslpath", "-u", out], text=True).strip())


def stage_for_windows(clip):
    """SoundPlayer wants a PCM WAV on a Windows path; convert and copy there."""
    audio, rate = sf.read(clip, dtype="int16")
    dst = windows_temp() / "mm_rec_play.wav"
    sf.write(dst, audio, rate, subtype="PCM_16")
    win = subprocess.check_output(["wslpath", "-w", str(dst)], text=True).strip()
    return win, len(audio) / rate


def start_playback(win_path):
    ps = f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"
    return subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", ps],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--secs", type=int, help="record length (default 5, or clip + 2 with --play)")
    ap.add_argument("--gain", type=int, default=14, help="ES7210 PGA gain 0-14 (~3dB/step)")
    ap.add_argument("--play", help="audio file to play through the laptop speaker while recording")
    ap.add_argument("--text", default="", help="what was said (for self-recorded clips)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    win_path = dur = None
    if args.play:
        win_path, dur = stage_for_windows(args.play)
    secs = args.secs or (math.ceil(dur + 2) if dur else 5)

    ser = open_port(args.port)
    wait_ready(ser)
    ser.write(f"rec {secs} {args.gain}\n".encode())

    chunks, stats, player = [], [], None
    t_start = None
    while True:
        raw = ser.readline()
        if not raw:
            if t_start and time.monotonic() - t_start > secs + 120:
                sys.exit("timeout waiting for recording data")
            continue
        line = raw.decode(errors="replace").rstrip()
        if line.startswith("@@REC start"):
            t_start = time.monotonic()
            print(line)
            if win_path:
                player = start_playback(win_path)
        elif line.startswith("@@A "):
            chunks.append(base64.b64decode(line[4:]))
        elif line.startswith("@@R "):
            stats.append(json.loads(line[4:]))
            print("  " + line[4:])
        elif line.startswith("@@DONE rec"):
            if not line.endswith("rc=0"):
                sys.exit(f"device error: {line}")
            break
        elif line.strip() and not line.startswith("bench>"):
            print(line)
    if player:
        player.wait(timeout=30)

    pcm = np.frombuffer(b"".join(chunks), dtype="<i2").reshape(-1, 2)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, pcm, RATE, subtype="PCM_16")
    meta = {"secs": secs, "gain": args.gain, "rate": RATE, "channels": ["mic1", "mic2"],
            "played": args.play, "text": args.text, "levels": stats,
            "recorded": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    peak = max(s["peak"] for s in stats) if stats else 0
    print(f"-> {out} ({len(pcm) / RATE:.1f}s stereo)")
    if peak >= 32000:
        print("warning: clipping -- lower --gain")
    elif peak < 1000:
        print("warning: very quiet -- raise --gain or check the mics")


if __name__ == "__main__":
    main()
