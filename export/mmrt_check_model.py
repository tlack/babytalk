"""End-to-end check of the mmrt runtime on the PC: run the exported model through the C
executor (mmrt/mmrt.c) and compare every intermediate tensor, the final logits and the
transcript with ESP-PPQ's simulation (data/models/mmrt/ref/*.npz).

  uv run mmrt_check_model.py [clip ...]
"""
import argparse
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import torch

from citrinet import ctc_greedy, load_model
from mmrt_info import MMRT, parse

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "mmrt" / "build" / "libmmrt.so"
MODEL = MMRT / "citrinet256_int8.mmrt"

TRACE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                         ctypes.c_int)
ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def build():
    LIB.parent.mkdir(exist_ok=True)
    src = ROOT / "mmrt"
    subprocess.check_call(["gcc", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC", "-o", str(LIB),
                           str(src / "mmrt.c"), str(src / "mmrt_ref.c")])
    return ctypes.CDLL(str(LIB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", default=["me-hello-world", "me-we-the-people", "1089-134686-0000"])
    a = ap.parse_args()
    lib = build()
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    alloc = ALLOC(lambda n: libc.malloc(n))
    free = FREE(lambda p: libc.free(p))

    g = parse()
    names = [g.nodes[0].inputs[0]] + [n.out for n in g.nodes]  # tensor id -> name (as exported)
    _, _, vocab, _ = load_model()
    image = MODEL.read_bytes()
    img_buf = ctypes.create_string_buffer(image, len(image))
    model = ctypes.create_string_buffer(64)
    assert lib.mmrt_open(model, img_buf, len(image)) == 0
    lib.mmrt_run.restype = ctypes.POINTER(ctypes.c_int8)
    lib.mmrt_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int), ALLOC, FREE]

    for clip in a.clips:
        ref = np.load(MMRT / "ref" / f"{clip}.npz", allow_pickle=True)
        state = {"checked": 0, "bad": []}

        def on_op(i, op_ptr, out, T, C, ref=ref, state=state):
            name = names[i + 1]
            if name not in ref.files:
                return
            y = np.ctypeslib.as_array(out, shape=(T * C,)).reshape(T, C)
            r = ref[name].reshape(T, -1)
            y = y[:, : r.shape[1]]
            state["checked"] += 1
            if not np.array_equal(y, r):
                state["bad"].append((i, name, int((y != r).sum())))

        cb = TRACE(on_op)
        ctypes.c_void_p.in_dll(lib, "mmrt_trace").value = ctypes.cast(cb, ctypes.c_void_p).value
        feats = np.ascontiguousarray(ref["feats"], dtype=np.int8)
        T_out = ctypes.c_int()
        out = lib.mmrt_run(model, feats.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), feats.shape[0],
                           ctypes.byref(T_out), alloc, free)
        assert out, "mmrt_run failed"
        logits = np.ctypeslib.as_array(out, shape=(T_out.value * 272,)).reshape(T_out.value, 272)[:, :257]
        ref_logits = ref[names[-1]].reshape(T_out.value, 257)
        text = ctc_greedy(torch.from_numpy(logits.T.astype(np.float32)), vocab)
        ref_text = ctc_greedy(torch.from_numpy(ref_logits.T.astype(np.float32)), vocab)
        exact = np.array_equal(logits, ref_logits)
        print(f"== {clip}: {state['checked']} tensors compared, {len(state['bad'])} differ; "
              f"logits {'BIT-EXACT' if exact else 'DIFFER'}")
        for i, name, nb in state["bad"][:5]:
            print(f"   op {i} {name}: {nb} values differ")
        print(f"   mmrt: {text!r}\n   ppq:  {ref_text!r}")
        ctypes.c_void_p.in_dll(lib, "mmrt_trace").value = None
    lib.mmrt_close(model, free)


if __name__ == "__main__":
    main()
