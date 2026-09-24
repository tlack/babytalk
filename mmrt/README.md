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
| 10.4 s clip | RTF 1.33 | 0.37 | **0.10** |
| 2 s command, model only | 2.5 s | ~1.0 s | **0.36 s** |
| app size | 1.8 MB | 1.8 MB | 0.9 MB |

## Speed log (model time, 2 cores, bit-exact at every step)

| step | 1 s | 2 s | 4 s | 10 s |
|---|---|---|---|---|
| first mmrt (staged 1x1, 2-core split) | 442 | 534 | | 1349 |
| + weight streaming for short inputs (core 0 copies, core 1 computes) | 374 | 496 | | |
| + fused depthwise -> 1x1 via per-core SRAM tiles | | | 675 | 1187 |
| + 6MB of weights cached in PSRAM at load (dw weights first) | 281 | 413 | 547 | 1041 |
| + activations <= 32KB in internal SRAM (40KB kept free) | **245** | **357** | **516** | **1015** |

Where the time goes now (`export/mmrt_profile.py`): the model is **memory-bound** at every
length. The 1x1 kernel itself runs at 1.19 cycles per VSMULAS (`c1bench`: ~3.2 GMAC/s
per core with operands in SRAM or cached PSRAM), but per inference 9.2MB of 1x1 weights
cross the shared flash/PSRAM bus, plus the activations. At 1 s the weight streamer (186
ms) is the critical path while core 1 computes ~140 ms; at 10 s staging is ~25% of the
1x1 wall. Copying on two cores instead of one does not help (bus-bound, flash or PSRAM).
Remaining levers: GDMA prefetch of the next op's weights from the PSRAM cache during
compute (GDMA can't read mapped flash), folding the SE mean into the producing kernel,
fewer weight bytes (int4 / pruning).

Why SRAM activations help short inputs: the streamer's flash reads go through the same
64KB dcache and evict core 1's PSRAM activations, which then miss behind the streamer
(depthwise at 1 s: 93 -> 37 ms). `internal_min_free` understates the low point here: it
sums each heap region's own minimum, reached at different times.

### 4-bit weights (CB4)

`mmrt.h` weight format per op: INT8 or CB4 (per 16-output group a 16-level int8 table per
channel + 4-bit indices), decoded to int8 wherever weights are staged, so kernels and
bit-exactness are unchanged. Model: GPTQ (act-order, bias correction) against each
channel's codebook, first 1x1 and decoder kept int8 (`export/int4_gptq.py --keep-io`,
packed by `export/mmrt_cb4.py`): **5.99 MB instead of 9.78 MB**, WER 8.21 vs 6.28 int8 on
test-clean (see `export/RESULTS.md`). Board output bit-exact with the host.

Decode in assembly (`mmrt_s3_cb4_rows`: per group a 1KB table of pre-shifted 32-bit
values, then extract / addx4 / load / OR per weight): 5.2 cycles per weight vs 9.3 for C.

| model time, ms | 1 s | 2 s | 4 s | 10 s |
|---|---|---|---|---|
| int8, 6MB PSRAM cache | 245 | 357 | 516 | 1015 |
| int8, no cache | 374 | 498 | 675 | 1187 |
| **int4, packed model cached (6MB)** | **241** | **325** | **512** | **998** |
| **int4, no cache** | **268** | **363** | **558** | **1071** |

Without a cache (PSRAM kept for the app, e.g. MicroPython), int4 is ~30% faster than
int8; half the bytes cross the flash bus.

### A dual-core race that was a kernel, not a race

A frame-loop depthwise kernel (`mmrt_s3_dw_rows`, all interior frames in one call, taps
software-pipelined two at a time) gave, on two cores only, one wrong 32-bit word in one
output row about once per 1000 calls: the same input produced different logits in ~1 of
10 ten-second runs. Tools that found it: `prof <T> 1 99999` (per-frame hashes of every
op, diffed across runs: always a single frame of a depthwise op) and `dwstress` (2-core
vs 1-core depthwise, thousands of iterations). Not the stack, not the QACC init, not
the rounding: the same loop was clean in 10k calls with a `nop` before the loop end or
unpipelined, and the proven per-frame `dw_row` path is just as fast end to end (bus-bound),
so the kernel was removed. Lesson: every kernel change gets a two-core `dwstress`-style
soak, not just `ktest`.
