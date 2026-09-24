#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial>=3.5"]
# ///
"""Drive the bench firmware over USB serial and collect results.

    tools/bench.py mem
    tools/bench.py fc 512 2048 16
    tools/bench.py all
    tools/bench.py --board tpager --port /dev/ttyACM1 membw

Each `@@R {json}` line the device prints is tagged with the run metadata and
appended to results/<date>-<board>.jsonl.
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

import serial

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDLE_TIMEOUT = 300  # seconds with no output before giving up


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def open_port(path):
    # USB-Serial-JTAG resets the chip on RTS/DTR toggles; keep both low.
    ser = serial.Serial()
    ser.port = path
    ser.baudrate = 115200
    ser.timeout = 1
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


def wait_ready(ser, timeout=10, prompt=b"bench>"):
    """Opening the port can reset the chip (USB-Serial-JTAG), so wait for the
    console prompt rather than assuming it's there. Pokes with newlines."""
    end = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < end:
        ser.write(b"\n")
        buf = (buf + ser.read(4096))[-4096:]
        if prompt in buf:
            time.sleep(0.2)
            ser.reset_input_buffer()
            return
    sys.exit(f"no {prompt.decode()} prompt within {timeout}s -- is the right firmware running?")


def run(ser, cmd, meta, out):
    name = cmd.split()[0]
    ser.write((cmd + "\n").encode())
    last = time.monotonic()
    count = 0
    while True:
        raw = ser.readline()
        if not raw:
            if time.monotonic() - last > IDLE_TIMEOUT:
                sys.exit(f"timeout: no output for {IDLE_TIMEOUT}s")
            continue
        last = time.monotonic()
        line = raw.decode(errors="replace").rstrip()
        if line.startswith("@@R "):
            rec = {**meta, **json.loads(line[4:])}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            count += 1
            print("  " + line[4:])
        elif line.startswith("@@DONE " + name + " "):
            rc = int(line.rsplit("rc=", 1)[1])
            print(f"{name}: rc={rc}, {count} results")
            return rc
        elif line.startswith("@@") or not line.startswith(("bench>", cmd)):
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--board", default="waveshare-s3cam")
    ap.add_argument("--note", default="", help="free-text tag stored with each result")
    ap.add_argument("cmd", nargs="+", help="bench command and its args")
    args = ap.parse_args()

    cmd = " ".join(args.cmd)
    now = datetime.datetime.now()
    meta = {
        "ts": now.isoformat(timespec="seconds"),
        "board": args.board,
        "cmd": cmd,
        "git": git_rev(),
        "note": args.note,
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    path = results / f"{now:%Y-%m-%d}-{args.board}.jsonl"

    ser = open_port(args.port)
    wait_ready(ser)
    with path.open("a") as out:
        rc = run(ser, cmd, meta, out)
    print(f"-> {path.relative_to(ROOT)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
