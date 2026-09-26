"""Profile the board's mmrt runtime op by op (firmware `prof` command).

Sends int8 model inputs of several lengths, gets per-op wall times, and reports time
by op kind, the 1x1-conv split between weight staging (flash -> SRAM) and compute,
effective GMAC/s per kernel, and the slowest ops.

  uv run mmrt_profile.py [--host IP] [--secs 2 4 10]
"""
import argparse
import collections
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np

from mmrt_check_model import MODEL
from mmrt_info import parse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from board_config import board_host  # noqa: E402

KIND = {1: "dwconv", 2: "conv1x1", 3: "mean", 4: "lut", 5: "mul", 6: "add"}


def load_ops():
    import struct
    img = MODEL.read_bytes()
    h = struct.unpack_from("<16I", img)
    n_t, n_ops, _, _, t_off, o_off = h[2], h[3], h[4], h[5], h[6], h[7]
    ch = [struct.unpack_from("<Hbx", img, t_off + 4 * i)[0] for i in range(n_t)]
    ops = []
    for i in range(n_ops):
        k, relu, K, stride, pad, shift, ea, eb, eo, in0, in1, out, w, b = struct.unpack_from(
            "<BBBBbbbbb3xHHHxxII", img, o_off + 28 * i)
        ops.append(dict(kind=KIND.get(k, "?"), K=K, stride=stride, cin=ch[in0], cout=ch[out]))
    return ops


def prof(host, T):
    s = socket.create_connection((host, 5555), timeout=600)
    f = s.makefile("rb")
    x = np.random.default_rng(0).integers(-60, 60, (T, 80)).astype(np.int8)
    s.sendall(f"prof {T}\n".encode() + x.tobytes())
    info, per_op = {}, []
    while True:
        line = f.readline().decode().strip()
        if line.startswith("DONE"):
            return info, per_op
        if line.startswith("OPS"):
            per_op = [tuple(map(int, p.split(":"))) for p in line.split()[1:]]
        elif line.startswith("{"):
            info.update(json.loads(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", help="board IP (default: $STT_HOST or board.conf)")
    ap.add_argument("--secs", type=float, nargs="+", default=[2, 4, 10])
    a = ap.parse_args()
    a.host = board_host(a.host)
    ops = load_ops()
    for secs in a.secs:
        T = int(secs * 100) + 1
        info, per_op = prof(a.host, T)
        total = sum(us for us, _ in per_op)
        print(f"\n=== {secs:.0f} s of audio (T={T}): model {info['model_ms']:.0f} ms (RTF {info['model_ms'] / 1000 / secs:.3f})")
        by = collections.defaultdict(lambda: [0, 0.0])  # us, MACs
        rows = []
        for i, ((us, To), op) in enumerate(zip(per_op, ops)):
            macs = 0
            if op["kind"] == "conv1x1":
                macs = To * op["cin"] * op["cout"]
            elif op["kind"] == "dwconv":
                macs = To * op["cin"] * op["K"]
            key = op["kind"] if op["kind"] in ("conv1x1", "dwconv", "mean") else "tail/other"
            by[key][0] += us
            by[key][1] += macs
            rows.append((us, i, op, To, macs))
        for k, (us, macs) in sorted(by.items(), key=lambda kv: -kv[1][0]):
            g = f"{macs / us / 1000:.2f} GMAC/s" if macs else ""
            print(f"  {k:12s} {us / 1000:7.1f} ms  {100 * us / total:5.1f}%  {g}")
        st, c1 = info.get("stage_us", [0, 0]), info.get("c1_us", [0, 0])
        print(f"  conv1x1 per core: staging {st[0] / 1000:.0f}/{st[1] / 1000:.0f} ms, compute {c1[0] / 1000:.0f}/{c1[1] / 1000:.0f} ms"
              f"  (core0/core1; staging = {100 * max(st) / max(1, by['conv1x1'][0]):.0f}% of conv1x1 wall)")
        print("  slowest ops:")
        for us, i, op, To, macs in sorted(rows, reverse=True)[:6]:
            g = f"{macs / us / 1000:.2f} GMAC/s" if macs else ""
            print(f"    #{i:3d} {op['kind']:8s} {op['cin']:3d}->{op['cout']:3d} K{op['K']:2d} s{op['stride']} T_out {To:4d}: {us / 1000:6.2f} ms {g}")


if __name__ == "__main__":
    main()
