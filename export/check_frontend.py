"""Check the C front end (stt/main/stt_core.c, host build stt/host_test/fe_test) against
the Python pipeline: normalized features, and the quantized static input window.

  gcc -O2 -o ../stt/host_test/fe_test ../stt/host_test/fe_test.c ../stt/main/stt_core.c -lm
  uv run check_frontend.py
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import evalwer
from citrinet import load_model, load_split
from recordings import load_recordings

FE = Path(__file__).resolve().parent.parent / "stt" / "host_test" / "fe_test"
EXP = -11  # w8a16 model input exponent


def main():
    _, feat, _, _ = load_model()
    items = [(u, p) for u, p, _ in load_recordings()]
    items += [(u, p) for u, p, _ in load_split()[::131]]  # 20 LibriSpeech clips
    worst_f, worst_q, bad_total, n_total = 0.0, 0, 0, 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for uid, path in items:
            a, sr = sf.read(str(path), dtype="int16")
            if a.ndim == 2:
                a = a[:, 0].copy()
            (td / "in.raw").write_bytes(a.astype("<i2").tobytes())
            T = int(subprocess.check_output([FE, td / "in.raw", td / "f.f32", td / "w.s16", str(EXP)]))
            c_feat = np.fromfile(td / "f.f32", dtype="<f4").reshape(T, 80)
            audio = torch.from_numpy(a.astype(np.float32) / 32768.0)
            py = feat(audio).numpy().T  # [T,80]
            df = float(np.abs(py - c_feat).max())
            line = f"{uid:34s} T={T:5d} feat maxdiff {df:.2e}"
            if T <= 1600:
                lm = feat.logmel(audio)[:, : feat.seq_len(len(audio))]
                win, _ = evalwer.fixed_windows(lm, fill="tile16")[0]
                q_py = np.clip(np.floor(win.numpy().T * 2.0 ** -EXP + 0.5), -32768, 32767).astype(np.int16)
                q_c = np.fromfile(td / "w.s16", dtype="<i2").reshape(1600, 80)
                d = np.abs(q_py.astype(int) - q_c.astype(int))
                bad_total += int((d > 0).sum())
                n_total += d.size
                worst_q = max(worst_q, int(d.max()))
                line += f"  int16 window: {int((d > 0).sum())} values differ (max {int(d.max())} LSB)"
            worst_f = max(worst_f, df)
            print(line)
    print(f"\nworst feature diff {worst_f:.2e}; int16 input: {bad_total}/{n_total} values differ "
          f"({100 * bad_total / max(n_total, 1):.4f}%), worst {worst_q} LSB")


if __name__ == "__main__":
    main()
