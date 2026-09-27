# MMRT: the micromodels runtime

MMRT runs the Citrinet speech recognizer on an ESP32-S3, and on an ESP32-P4 (see [Other
chips](#other-chips-the-esp32-p4)). It is a few thousand lines of C and assembly: a model file
format, an executor, portable reference operations, and hand-written kernels for the
ESP32-S3's vector instructions (PIE) and the ESP32-P4's. It runs any model
built from the same handful of operations: 1D depthwise and pointwise (1x1)
convolutions, mean, lookup tables, multiply and add. Citrinet and similar convolutional
speech models fit that description.

## Why not ESP-DL?

Espressif's [ESP-DL](https://github.com/espressif/esp-dl) was the first thing we tried,
and it produced correct results. On this model it was too slow, and making it faster
meant patching it:

| | ESP-DL, stock | ESP-DL, our patches | MMRT |
|---|---|---|---|
| 10.4 s clip (time per second of audio) | 1.33 s | 0.37 s | **0.10 s** |
| 2 s command | 2.5 s | ~1.0 s | **0.36 s** |
| app size | 1.8 MB | 1.8 MB | 0.9 MB |

Stock ESP-DL re-read each 1x1 convolution's weights from flash for every output frame,
and it ran 1D convolutions on one core only. We had patched both, but a patched vendor
library is hard for anyone else to build and maintain. A small runtime of our own was
simpler, and it opened up the optimizations below. The outputs didn't change: MMRT gives
the same int8 results as ESP-DL, bit for bit.

## What makes it fast

The chip can do far more arithmetic than its memory can feed. Flash delivers ~32 MB/s and
PSRAM ~88 MB/s, and they share one bus; internal SRAM is 5-10x faster but small. So
nearly every speed-up here is about moving fewer bytes:

- **Weights are read once per layer, not once per frame.** Each 1x1 layer's weights are
  copied into a 32 KB SRAM buffer per core, then applied to every frame of audio.
- **Both cores work**, each on half the output channels. The two staging buffers sit in
  different SRAM banks, because two cores reading one bank slow each other down
  (1.17x vs 1.6x speed-up).
- **Short clips stream the weights.** For short clips (under ~2.5 s) core 0 does nothing
  but copy the next layers' weights while core 1 computes, so the copying overlaps the math.
- **Layers are fused** so intermediate results stay in SRAM. A depthwise convolution
  feeds its 1x1 convolution through a 16-frame SRAM tile. A block's scale, residual add
  and ReLU run as one pass.
- **Vector kernels in assembly.** The 1x1 kernel does 16 multiply-adds per instruction at
  1.19 CPU cycles per instruction, about 3.2 billion multiply-adds per second per core.
- **4-bit weights** halve the bytes read from flash. They are unpacked to 8-bit while being
  copied into SRAM, by an assembly routine at 5.2 cycles per weight, so the math kernels
  never change.
- **Optional PSRAM weight cache**: up to 6 MB of weights in PSRAM (2.75x faster than flash).

Everything is checked for exactness: each kernel against the portable C reference on
random data, and the whole model against a PC run of the same file.

## Model files and weight formats

A `.mmrt` file holds a header, a tensor table, a list of 28-byte operations, and a
16-byte-aligned block of weights (`mmrt.h`). The runtime reads it **in place** from
memory-mapped flash; only activations are allocated. Each 1x1 layer's weights are one of:

- **INT8**: 8-bit weights.
- **CB4**: a 4-bit codebook. Each output channel has a table of 16 8-bit values, and every
  weight is a 4-bit index into its channel's table. 256 + 8 x C bytes per 16 channels
  instead of 16 x C.

The format is recorded per layer, so int8, int4 and mixed files all load the same way.
Files are produced by `export/mmrt_export.py` (int8), then `export/int4_gptq.py` and
`export/mmrt_cb4.py` (int4). Ready-made files are in `../models/`.

## Using it from C

```c
#include "mmrt.h"
#include "mmrt_s3.h"

mmrt_model_t m;
mmrt_open(&m, image, image_size);   // image: memory-mapped .mmrt (e.g. esp_partition_mmap)
mmrt_s3_init();                     // SRAM staging buffers + the core-0 worker task
int T_out;
const int8_t *logits = mmrt_run(&m, features, T_in, &T_out, alloc, release);  // [T_out][272]
```

`alloc`/`release` supply activation memory (PSRAM on the S3). `mmrt_cache_weights()`
enables the PSRAM cache. Build with `MMRT_S3` defined for the S3 kernels; without it, add
`mmrt_port.c` and you get the portable kernels (the reference's arithmetic, laid out for
speed), which is how the PC tools run the same file. `MMRT_P4` adds the ESP32-P4's vector
kernels (`mmrt_p4.S`, see below). The complete speech-to-text pipeline around it (audio
features, decoding, wake phrases) is `components/stt_engine/`, and the ESP-IDF component
`components/mmrt` picks the right files for the target chip.

| file | what |
|---|---|
| `mmrt.h`, `mmrt.c` | file format, API, executor (sizes tensors per input length, frees them at last use, fuses layers) |
| `mmrt_ref.c/.h` | portable reference operations: the definition of correct |
| `mmrt_s3.S` | PIE kernels: 1x1 conv, depthwise conv, fused block tail, column sums, CB4 decode |
| `mmrt_s3.c/.h` | drivers: weight staging, two-core split, streaming, fusion, buffers |
| `mmrt_port.c/.h` | portable kernels for other chips, bit-exact with the reference; on the P4 also the drivers for its vector kernels (two-core split, self-checks) |
| `mmrt_p4.S` | the ESP32-P4's vector kernels: int8 dot product, depthwise conv, one 1x1 group from the packed layout (the `mmrt` NIFs' matmul) |

## Results (Waveshare ESP32-S3-CAM, ESP-IDF app, milliseconds of model time)

| model | 1 s | 2 s | 4 s | 10 s |
|---|---|---|---|---|
| int8, no cache | 374 | 498 | 675 | 1187 |
| int8, 6 MB PSRAM cache | 245 | 357 | 516 | 1015 |
| int4, no cache | 268 | 363 | 558 | 1071 |
| int4, whole model cached | 241 | 325 | 512 | 998 |

Under MicroPython (`mpy/`) the same model runs 10-17% slower. The likely cause: there the
SRAM staging buffers share one memory block with the text-to-speech engine, so both
cores read the same bank. Numbers are in the top-level README.

## Engineering notes

**Tests.** `export/mmrt_check_ops.py` checks every reference op against ESP-PPQ's
simulation (368/368 bit-exact), and `export/mmrt_check_model.py` checks the whole model on
the PC. The firmware's `ktest` checks every PIE kernel against its reference, and
`export/mmrt_check_device.py` compares the board with the PC on the same input.
`dwstress` runs two-core kernels thousands of times against one core.

**Where the time goes** (`export/mmrt_profile.py`): the model is memory-bound at every
length. Per int8 inference, 9.2 MB of 1x1 weights cross the shared flash/PSRAM bus, plus
the activations. At 1 s of audio the weight copying (~186 ms) is the critical path while
the other core computes ~140 ms.

**Small facts that shaped the kernels.** Every ReLU lookup table in this model is an exact
ReLU, and every residual add has equal input scales, so a whole block tail is
`max(0, sat8(round(a*s >> sh) + r))`. The largest accumulator value is 142,451, safely
inside the vector unit's 20-bit accumulator lanes.

**Speed history** (model time in ms, 2 cores, bit-exact at every step, int8):

| step | 1 s | 2 s | 4 s | 10 s |
|---|---|---|---|---|
| first MMRT (staged 1x1, 2-core split) | 442 | 534 | | 1349 |
| + weight streaming for short clips | 374 | 496 | | |
| + fused depthwise -> 1x1 | | | 675 | 1187 |
| + 6 MB PSRAM weight cache | 281 | 413 | 547 | 1041 |
| + small activations in internal SRAM | 245 | 357 | 516 | 1015 |

**A two-core bug that was really a kernel bug.** A depthwise kernel that processed all
frames in one call with a software-pipelined loop produced one wrong 32-bit word about
once per 1,000 calls, only with both cores running. The same loop with a `nop` before
its end, or without pipelining, was clean in 10,000 calls. It was no faster end to end,
so it was removed. Since then every kernel change gets a two-core soak test, not just the
single-shot `ktest`.

## Other chips: the ESP32-P4

On a chip without the S3's vector unit, MMRT runs **portable kernels** (`mmrt_port.c`). These
are the reference's arithmetic, bit for bit, laid out for speed: sixteen outputs at a time from
the packed weights, and zero inputs (after a ReLU, most of them) skipped.

The **ESP32-P4** has a vector unit of its own (the `Xesppie` RISC-V extension, 128-bit
registers). MMRT uses it as follows (`MMRT_P4`, set by `components/mmrt` for `esp32p4`):

- **1x1 convolutions a 16-output group at a time**, straight from the packed layout: each input
  times its 16 weights in one `esp.vsmulas.s8.qacc`, which also loads the weights two steps
  ahead (the S3's loop). The bias rides along as one more 16-input chunk: constant inputs
  {1, 64, 64, ...} and a block of weights that sum to it exactly (any bias up to ~122,000). The
  frames go 32 at a time, so a tile stays in cache while every group meets it, and the two
  cores take half the frames each. int4 layers are decoded once and kept in PSRAM.
- **int4 decoding** on the vector unit: there's no gather, so per row the 16 lane indices are
  unpacked (mask, shift, `esp.vzip.8`) and each of the 16 codebook levels is compared against
  every lane at once (`esp.vcmp.eq.u8`, AND, OR).
- **Row dot products** (`esp.vmulas.s8.xacc`, a 40-bit accumulator) for anything the group
  kernel can't take: a bias too large, a shift over 13, or the `mmrt` NIFs' exact int32 results.
- **Depthwise convolutions** on `esp.vmulas.s8.qacc`, 16 channels per instruction: the S3's
  kernel, ported.
- Each vector kernel **checks itself** against the portable one on a test layer before its
  first use, and stays unused if they differ.
- Activations go to **internal RAM** when they fit (`components/stt_engine`).

The host checker runs the P4's paths in C: `MMRT_CFLAGS="-DMMRT_C1 -DMMRT_ROWDOT" uv run
export/mmrt_check_model.py` (the group kernel's arithmetic, biases and tiling included: 164 of
Citrinet-256's 165 1x1 layers take it) is bit-exact on the whole model.

**Speed history on the P4** (transcribing 2 s of speech, int4 model, bit-exact at every step):

| step | 2 s | 8 s |
|---|---|---|
| reference kernels (`mmrt_ref.c`) | 14.4 s | |
| portable kernels | 8.7 s | |
| + 1x1 as vector dot products, rows kept | 2.7 s | |
| + depthwise on the vector unit | 1.25 s | |
| + 1x1 on both cores | 0.91 s | |
| + activations in internal RAM | 0.83 s | 30 s |
| + frames in tiles of 32 | 0.87 s | 3.6 s |
| + 1x1 on the 16-lane group kernel, bias as a chunk | 0.65 s | 2.8 s |
| + int4 decoded on the vector unit (the first run: 1.2 s -> 1.06 s) | 0.65 s | 2.7 s |
| + the S3's fused load-and-multiply loop | 0.58 s | 2.6 s |
| PSRAM at 200 MHz, not ESP-IDF's default 20 (every step above was measured at 20) | 0.37 s | 1.37 s |
| + block tails fused on the vector unit (`mmrt_p4_tail_row`) | **0.30 s** | **1.10 s** |

The S3 does 2 s in 0.53 s. Not yet on the P4: the depthwise -> 1x1 fusion the S3 uses on short
clips.

**The `mmrt` NIFs on the P4** (256x256, the same kernels): int8 matvec 267 MMAC/s, matmul with
16 rows 1640 MMAC/s, int4 564 MMAC/s. The S3: 145, 720 and 466. The P4's own quirks (saturation, its zero-overhead
loop, registers, RTC RAM) are in `docs/BOARD_WAVESHARE_P4_WIFI6.md`.
