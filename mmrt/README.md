# mmrt — micromodels runtime

A small int8 inference runtime for Citrinet-style CTC speech models on the ESP32-S3,
replacing ESP-DL. Portable C reference ops define the arithmetic; PIE (SIMD) assembly
kernels replace them op by op, each proven bit-exact before use.

| file | what |
|---|---|
| `mmrt.h` | model image format (header, tensors, 28-byte ops, 16B-aligned weight blob) + API |
| `mmrt.c` | executor: sizes tensors per input length, frees at last use, dispatches kernels, fuses block tails |
| `mmrt_ref.c/.h` | reference ops: exact result, round half up at the output exponent, saturate |
| `mmrt_s3.S` | PIE kernels: 1x1 conv, depthwise conv, fused block tail, column sums |
| `mmrt_s3.c/.h` | kernel drivers: SRAM weight staging, two-core split, QACC bias packing |

## Pipeline and checks

```
export/mmrt_quant.py        ESP-PPQ int8 quantization + per-op reference activations (3 clips)
export/mmrt_info.py         parse the deployed graph; verify weight layouts vs ONNX floats
export/mmrt_check_ops.py    each reference op vs ESP-PPQ, node by node (bit-exact: 368/368)
export/mmrt_export.py       -> data/models/mmrt/citrinet256_int8.mmrt (9.78 MB)
export/mmrt_check_model.py  whole model on the PC vs ESP-PPQ (bit-exact, every tensor)
stt firmware: ktest         each PIE kernel vs its reference op on random data (28 cases)
export/mmrt_check_device.py board vs PC on the same input (logits FNV)
```

Flash the model: `esptool.py write_flash 0x410000 data/models/mmrt/citrinet256_int8.mmrt`.

## Kernels (ESP32-S3 @ 240 MHz)

| kernel | idea | speed |
|---|---|---|
| 1x1 conv | broadcast MAC `EE.VSMULAS.S8.QACC.LD.INCP` (16 MAC/instr), bias preloaded into QACC, round-half-up via two SRCMB shifts; weights staged into per-core 32KB SRAM buffers | 2.6 GMAC/s one core, 4.2 two cores (PIE peak 3.84/core) |
| depthwise conv | lane-wise MAC down the time axis (`VMULAS.LD.IP` + `VLD.XP`), taps clipped in C | ~1 GMAC/s/core, scales 2x |
| block tail (fused) | SE scale (QACC + rounding) -> saturating residual add -> ReLU, one pass | 13x the three C ops |
| mean over time | column sums in QACC (x * ones), exact division in C | 15x |

Findings that shaped the design:
- ESP-DL re-streamed each 1x1 filter from flash per output frame and never split 1D
  convs across cores; staging weights in SRAM once per op was worth ~3x on its own.
- Two cores reading weights from the same SRAM bank stall each other (1.17x). Separate
  per-core buffers in different SRAM regions: 1.6x.
- All ReLU LUTs in this model are identity-ReLU and all residual adds have equal
  exponents, so each block tail is `max(0, sat8(round(a*s >> sh) + r))`.
- Largest accumulator 142451 < 2^19: the 20-bit QACC lanes never wrap on this model.
- Per inference, 9.8MB of weights come out of flash (~0.3 s): the floor for short clips.

## Results (Waveshare S3 cam, int8, same outputs as ESP-DL and the ESP-PPQ simulation)

| | stock ESP-DL | patched ESP-DL (2 cores) | mmrt (2 cores) |
|---|---|---|---|
| 10.4 s clip | RTF 1.33 | 0.37 | **0.13** |
| 2 s command, model only | 2.5 s | ~1.0 s | **0.53 s** |
| app size | 1.8 MB | 1.8 MB | 0.9 MB |

Next: the float log-mel front end (now ~40% of total time; can run while recording),
weight-staging overlap / faster flash, int8 accuracy via QAT.
