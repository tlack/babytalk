"""Quantize the static Citrinet ONNX for ESP-DL (esp32s3) with ESP-PPQ, export .espdl, and
evaluate WER of the quantized graph simulated on the host (PPQ TorchExecutor).

  uv run quantize.py                       # int8 (w8a8), 128 dev-clean calibration windows
  uv run quantize.py --mixed               # + first conv(s) and CTC decoder conv in int16
  uv run quantize.py --every 4             # eval on every 4th test-clean utt (default; 655 utts)

Calibration uses dev-clean only; evaluation uses test-clean, so they never overlap.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time

import torch

import evalwer
from citrinet import DATA, load_model, load_split, read_audio
from evalwer import RESULTS, WIN, fixed_windows, score

from esp_ppq.api import espdl_quantize_onnx, get_target_platform  # noqa: E402
from esp_ppq.api.setting import QuantizationSettingFactory  # noqa: E402
from esp_ppq.core import TargetPlatform  # noqa: E402
from esp_ppq.executor import TorchExecutor  # noqa: E402

MODELS = DATA / "models"


def calib_windows(feat, n: int, split="dev-clean"):
    """n [1,80,WIN] tiled windows from utterances spread evenly over the split (first window
    of each; long clips contribute their first 16 s, normalized on their own)."""
    items = load_split(split)
    step = len(items) / n
    out = []
    for i in range(n):
        _, p, _ = items[int(i * step)]
        a = read_audio(p)
        lm = feat.logmel(a)[:, : feat.seq_len(a.shape[0])]
        x, _ = fixed_windows(lm[:, :WIN], fill="tile16")[0]
        out.append(x[None])
    return out


def input_exponent(graph, name="feats"):
    var = graph.variables[name]
    for op in var.dest_ops:
        cfg = op.input_quant_config[op.inputs.index(var)]
        s = float(cfg.scale.flatten()[0])
        return s, math.log2(s)
    return None, None


def exponents(graph):
    """(op name, [input exps], [output exps]) for every op, from the quantized graph."""
    rows = []
    for op in graph.operations.values():
        ie = [round(math.log2(float(c.scale.flatten()[0]))) if c.scale is not None and c.scale.numel() == 1 else None
              for c in op.input_quant_config]
        oe = [round(math.log2(float(c.scale.flatten()[0]))) if c.scale is not None and c.scale.numel() == 1 else None
              for c in op.output_quant_config]
        rows.append((op.name, op.type, ie, oe))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--mixed", action="store_true", help="int16 for block-0 convs + decoder conv")
    ap.add_argument("--int16", action="store_true", help="whole graph w16a16")
    ap.add_argument("--every", type=int, default=4, help="eval on every Nth test-clean utterance")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    tag = "int16" if a.int16 else ("int8_mixed16" if a.mixed else "int8")
    src = MODELS / "citrinet256_static1600.onnx"
    work = MODELS / "ppq"
    work.mkdir(parents=True, exist_ok=True)
    onnx_in = work / f"citrinet256_static1600_{tag}.onnx"  # ESP-PPQ simplifies this file in place
    shutil.copy(src, onnx_in)
    espdl = MODELS / f"citrinet256_s3_{tag}.espdl"

    m, feat, vocab, _ = load_model()
    t0 = time.time()
    calib = calib_windows(feat, a.calib)
    print(f"calibration: {len(calib)} dev-clean windows ({time.time() - t0:.0f}s)")

    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=16 if a.int16 else 8)
    if a.mixed:
        for op in ["/blocks/blocks.0/convs.0/convs.0.0/Conv", "/blocks/blocks.0/convs.0/convs.0.1/Conv",
                   "/decoder/Conv"]:
            setting.dispatching_table.append(op, get_target_platform("esp32s3", 16))

    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_in), espdl_export_file=str(espdl), calib_dataloader=calib,
        calib_steps=len(calib), input_shape=[1, 80, WIN], target="esp32s3", num_of_bits=16 if a.int16 else 8,
        setting=setting, device="cpu", error_report=False, export_test_values=False, verbose=0,
    )
    qsecs = time.time() - t0
    scale, exp = input_exponent(graph)
    print(f"quantized+exported in {qsecs:.0f}s -> {espdl} ({espdl.stat().st_size / 1e6:.2f} MB)")
    print(f"input 'feats' scale {scale} exponent {exp}")
    plat = {}
    for op in graph.operations.values():
        plat[str(op.platform)] = plat.get(str(op.platform), 0) + 1
    print("op platforms:", plat)

    bad = []
    with open(RESULTS / f"exponents_{tag}.tsv", "w") as f:
        for name, typ, ie, oe in exponents(graph):
            f.write(f"{name}\t{typ}\t{ie}\t{oe}\n")
            if typ in ("Conv", "Gemm") and None not in ie[:2] and oe[0] is not None and oe[0] - ie[0] - ie[1] < 0:
                bad.append((name, ie, oe))
    print("ops violating out_exp >= in_exp + w_exp:", bad)
    summ = dict(tag=tag, exponent_violations=[b[0] for b in bad], espdl=str(espdl), espdl_bytes=espdl.stat().st_size, input_scale=scale,
                input_exponent=exp, calib=len(calib), platforms=plat)

    if not a.no_eval:
        ex = TorchExecutor(graph=graph, device="cpu")

        @torch.no_grad()
        def runner(x):
            return ex.forward(inputs=x)[0][0].float()

        # float reference on the same subset, same static pipeline
        items = load_split()[:: a.every]
        evalwer._G.update(m=m, feat=feat, vocab=vocab, mode="fixed", fill="tile16", long="overlap")
        t0 = time.time()
        rows_f = [evalwer._one(it) for it in items]
        evalwer._G["runner"] = runner
        rows_q = []
        for i, it in enumerate(items):
            rows_q.append(evalwer._one(it))
            if i % 100 == 0:
                print(f"  {i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        del evalwer._G["runner"]
        wf, wq = score(rows_f), score(rows_q)
        summ.update(n=len(items), wer_float_static=round(wf, 3), wer_quant=round(wq, 3),
                    eval_secs=round(time.time() - t0, 1))
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / f"ppq_{tag}_every{a.every}.tsv", "w") as f:
            for uid, ref, hyp, dur in rows_q:
                f.write(f"{uid}\t{dur:.2f}\t{ref}\t{hyp}\n")
    with open(RESULTS / "quant_summary.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
