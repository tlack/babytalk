"""Send WAV clips to the board's stt_server (atomvm/apps/stt_server) over WiFi TCP and print
its transcripts, next to the host int4 transcript from a field_eval run when there is one.

    python3 atomvm/tools/stt_client.py clip.wav ...
    python3 atomvm/tools/stt_client.py --field 6      # 6 field clips with host int4 transcripts
    python3 atomvm/tools/stt_client.py --field 8 --save atomvm.json   # keep the transcripts

The board's address: --host, else $STT_HOST, else board.conf at the repo root (see
board.conf.example).

WAVs must be 16 kHz int16; channel 0 of a multichannel file is sent (as the board's engine does).
"""
import argparse
import csv
import glob
import json
import socket
import struct
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from board_config import board_host  # noqa: E402


def pcm_of(path):
    w = wave.open(str(path))
    assert w.getframerate() == 16000 and w.getsampwidth() == 2, f"{path}: need 16 kHz int16"
    raw, ch = w.readframes(w.getnframes()), w.getnchannels()
    if ch == 1:
        return raw
    return b"".join(raw[i:i + 2] for i in range(0, len(raw), 2 * ch))


def field_clips(n):
    """Clips from the newest field_eval TSV with an int4 column whose WAV still exists."""
    man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/field/manifest.jsonl")}
    for tsv in sorted(glob.glob(str(ROOT / "data/results/field_eval_*.tsv")), reverse=True):
        rows = [r for r in csv.DictReader(open(tsv), delimiter="\t") if r.get("int4") is not None]
        rows = [(ROOT / "data/field" / man[r["id"]]["file"], r) for r in rows if r["id"] in man]
        rows = [(p, r) for p, r in rows if p.exists()]
        if rows:
            step = max(1, len(rows) // n)
            return rows[::step][:n]
    sys.exit("no field_eval TSV with existing clips")


def connect(host, port):
    for _ in range(30):  # the board may still be booting
        try:
            s = socket.create_connection((host, port), timeout=120)
            return s, s.makefile("rb")
        except OSError:
            time.sleep(2)
    sys.exit(f"cannot connect to {host}:{port}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="board IP (default: $STT_HOST or board.conf)")
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--field", type=int, default=0, metavar="N")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--save", metavar="JSON", help="write {clip file name: board transcript} here")
    a = ap.parse_args()
    a.host = board_host(a.host)
    clips = [(Path(p), None) for p in a.wavs] + (field_clips(a.field) if a.field else [])
    s = f = None
    same, saved = 0, {}
    for path, row in clips:
        pcm = pcm_of(path)
        for attempt in range(2):  # one resend: AtomVM's gen_tcp occasionally drops a long upload
            if s is None:
                s, f = connect(a.host, a.port)
            t0 = time.time()
            try:
                s.sendall(struct.pack(">I", len(pcm)) + pcm)
                line = f.readline()
                if line:
                    break
            except OSError:
                pass
            print(f"  (connection dropped, {'resending' if attempt == 0 else 'giving up'})")
            s.close()
            s = None
        else:
            continue
        reply = line.decode().rstrip("\n").split("\t")
        print(f"{path.name}  {len(pcm) / 32000:.1f} s audio, {time.time() - t0:.2f} s round trip")
        if reply[0] != "ok":
            print(f"  board ERROR: {reply}")
            continue
        text, ms, info = reply[1], reply[2], reply[3]
        saved[path.name] = text
        print(f"  board:     {text!r}  ({ms} ms; {info})")
        if row:
            print(f"  host int4: {row['int4']!r}\n  ref:       {row['ref']!r}")
            same += text == row["int4"]
    if a.save:
        Path(a.save).write_text(json.dumps(saved, indent=1))
    if any(r for _, r in clips):
        print(f"board == host int4 on {same}/{sum(1 for _, r in clips if r)} clips")


if __name__ == "__main__":
    main()
