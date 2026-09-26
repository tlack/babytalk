"""Bit-exact check of the board's mmrt runtime against the host build of the same C code.

Sends each reference clip's int8 features (data/models/mmrt/ref/*.npz) to the board
(`feats` command, WiFi) and compares the FNV-1a of the valid logits [T_out][257] with
the host run; prints the board's per-op-kind timing. Run after every kernel change.

  uv run mmrt_check_device.py [--host IP] [clip ...]
"""
import argparse
import ctypes
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np

from mmrt_check_model import ALLOC, FREE, MODEL, build
from mmrt_info import MMRT
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from board_config import board_host  # noqa: E402


def fnv(y, V):
    h = 2166136261
    for b in np.ascontiguousarray(y[:, :V]).view(np.uint8).ravel().tolist():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return f"{h:08x}"


def host_logits(lib, model, feats, alloc, free):
    T_out = ctypes.c_int()
    out = lib.mmrt_run(model, feats.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), feats.shape[0],
                       ctypes.byref(T_out), alloc, free)
    return np.ctypeslib.as_array(out, shape=(T_out.value * 272,)).reshape(T_out.value, 272).copy()


def board_feats(host, feats):
    s = socket.create_connection((host, 5555), timeout=600)
    f = s.makefile("rb")
    s.sendall(f"feats {feats.shape[0]}\n".encode() + feats.tobytes())
    info = {}
    while True:
        line = f.readline().decode().strip()
        if line.startswith("DONE"):
            info["rc"] = int(line.rsplit("rc=", 1)[1])
            return info
        if line.startswith("{"):
            info.update(json.loads(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", help="board IP (default: $STT_HOST or board.conf)")
    ap.add_argument("--model", default=str(MODEL), help="the image flashed to the board's model partition")
    ap.add_argument("clips", nargs="*", default=["me-hello-world", "me-we-the-people", "1089-134686-0000"])
    a = ap.parse_args()
    a.host = board_host(a.host)
    lib = build()
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    alloc, free = ALLOC(lambda n: libc.malloc(n)), FREE(lambda p: libc.free(p))
    image = open(a.model, "rb").read()
    buf = ctypes.create_string_buffer(image, len(image))
    model = ctypes.create_string_buffer(256)
    assert lib.mmrt_open(model, buf, len(image)) == 0
    lib.mmrt_run.restype = ctypes.POINTER(ctypes.c_int8)
    lib.mmrt_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int), ALLOC, FREE]
    ok = True
    for clip in a.clips:
        feats = np.ascontiguousarray(np.load(MMRT / "ref" / f"{clip}.npz")["feats"], dtype=np.int8)
        want = fnv(host_logits(lib, model, feats, alloc, free), 257)
        got = board_feats(a.host, feats)
        same = got.get("logits_fnv") == want
        ok &= same
        audio_s = feats.shape[0] / 100
        ops = " ".join(f"{k} {v / 1000:.2f}s" for k, v in sorted(got.get("ops_ms", {}).items(), key=lambda kv: -kv[1]))
        print(f"{clip:20s} {'BIT-EXACT' if same else 'MISMATCH '} host {want} board {got.get('logits_fnv')}  "
              f"model {got.get('model_ms', 0) / 1000:.2f}s for {audio_s:.1f}s audio "
              f"(RTF {got.get('model_ms', 0) / 1000 / audio_s:.2f})\n{'':22s}{ops}\n{'':22s}{got.get('text')!r}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
