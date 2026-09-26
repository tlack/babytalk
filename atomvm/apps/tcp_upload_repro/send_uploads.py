"""Laptop side of tcp_upload_repro: send N length-prefixed uploads of random size to the board
and count how many the board fails to receive (connection reset/closed before its "ok").

    python3 send_uploads.py BOARD_IP [-n 40] [--min 100000] [--max 330000] [--gap 1.0]

--gap: seconds to wait between uploads (our app did ~1.3 s of work on the board between them).
After a failure it reconnects and carries on.
"""
import argparse, os, random, socket, struct, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("host")
ap.add_argument("-n", type=int, default=40)
ap.add_argument("--port", type=int, default=5556)
ap.add_argument("--min", type=int, default=100_000)
ap.add_argument("--max", type=int, default=330_000)
ap.add_argument("--gap", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = random.Random(a.seed)

def connect():
    s = socket.create_connection((a.host, a.port), timeout=30)
    return s, s.makefile("rb")

s, f = connect()
fails = 0
for i in range(1, a.n + 1):
    size = rng.randint(a.min, a.max)
    payload = os.urandom(size)
    t0 = time.time()
    try:
        s.sendall(struct.pack(">I", size) + payload)
        line = f.readline().decode().strip()
        ok = line == f"ok {size}"
    except OSError as e:
        line, ok = f"{type(e).__name__}: {e}", False
    ms = (time.time() - t0) * 1000
    print(f"upload {i:3d}: {size:7d} bytes  {'ok  ' if ok else 'FAIL'}  {ms:6.0f} ms  {line if not ok else ''}", flush=True)
    if not ok:
        fails += 1
        s.close()
        time.sleep(1)
        s, f = connect()
    time.sleep(a.gap)
s.close()
print(f"{fails}/{a.n} uploads failed")
sys.exit(1 if fails else 0)
