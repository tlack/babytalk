"""Quantize Citrinet-256 int8 for the mmrt runtime and save the ESP-PPQ graph.

Same settings as the board's current model (quantize.py --tag cle_int8): the CLE ONNX,
ESP-PPQ esp32s3 int8, 128 dev-clean calibration windows. The graph is saved so the
exporter and the layer-by-layer checks can reload it in seconds.

  uv run mmrt_quant.py      -> data/models/mmrt/citrinet256_int8.{espdl,info,json}
                               data/models/mmrt/ref/<clip>.npz

The .info text dump is the deployed graph (ops, weights in kernel layout, every
tensor's exponent): mmrt_export.py builds the runtime model from it. The ref/*.npz
files are ESP-PPQ's simulated activations for a few clips at their exact lengths --
every intermediate tensor as int8, time-major [T, C] -- the per-op ground truth for
checking the mmrt kernels.
"""
import shutil
import time

import numpy as np
import torch
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.core import QuantizationStates
from esp_ppq.executor import TorchExecutor

from citrinet import DATA, load_model, read_audio
from recordings import REC
from quantize import MODELS, WIN, calib_windows, input_exponent

OUT = MODELS / "mmrt"
REF_CLIPS = [REC / "me-hello-world.wav", REC / "me-we-the-people.wav",
             DATA / "librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"]


def dump_reference(graph, feat, clip, out):
    """Every activation of the quantized graph for one clip, as int8 [T, C]."""
    audio = read_audio(clip)
    x = feat(audio)[None]  # normalized features [1, 80, T]
    acts = [n for n, v in graph.variables.items() if not v.is_parameter]
    ex = TorchExecutor(graph=graph, device="cpu")
    vals = ex.forward(inputs=x, output_names=acts)
    arrays, exps = {}, {}
    for name, v in zip(acts, vals):
        op = graph.variables[name].source_op
        cfg = op.output_quant_config[op.outputs.index(graph.variables[name])] if op else None
        if cfg is None:  # graph input: raw float here, quantized inside its first consumer
            dest = graph.variables[name].dest_ops[0]
            cfg = dest.input_quant_config[dest.inputs.index(graph.variables[name])]
            scale = float(cfg.scale.flatten()[0])
            v = torch.clamp(torch.floor(v / scale + 0.5), -128, 127) * scale  # round half up, as stt_core.c
        if not QuantizationStates.is_activated(cfg.state):
            continue  # fused away on deployment (e.g. a conv output feeding a Relu)
        scale = float(cfg.scale.flatten()[0])
        q = torch.round(v / scale).to(torch.int32)
        assert q.abs().max() <= 128 and torch.allclose(q * scale, v, atol=scale * 1e-3), name
        arrays[name] = q[0].T.contiguous().numpy().astype(np.int8)  # [C, T] -> [T, C]
        exps[name] = int(round(np.log2(scale)))
    np.savez_compressed(out, **arrays, __exponents__=np.array(list(exps.items()), dtype=object))
    return len(arrays)


def main():
    torch.set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)
    onnx_in = OUT / "citrinet256_static1600_cle.onnx"  # ESP-PPQ simplifies this in place
    shutil.copy(MODELS / "citrinet256_static1600_cle.onnx", onnx_in)
    _, feat, _, _ = load_model()
    calib = calib_windows(feat, 128)
    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_in), espdl_export_file=str(OUT / "citrinet256_int8.espdl"),
        calib_dataloader=calib, calib_steps=len(calib), input_shape=[1, 80, WIN], target="esp32s3",
        num_of_bits=8, setting=QuantizationSettingFactory.espdl_setting(num_of_bits=8), device="cpu",
        error_report=False, export_test_values=False, verbose=0,
    )
    print(f"quantized in {time.time() - t0:.0f}s; input {input_exponent(graph)}")
    (OUT / "ref").mkdir(exist_ok=True)
    for clip in REF_CLIPS:
        out = OUT / "ref" / f"{clip.stem}.npz"
        n = dump_reference(graph, feat, clip, out)
        print(f"-> {out.name}: {n} tensors")


if __name__ == "__main__":
    main()
