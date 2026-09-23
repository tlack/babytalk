# Phase 0 results

Board: Waveshare S3 cam `sentry-6aa8` (ESP32-S3 rev v0.2, 240MHz, 16MB QIO flash
@80MHz, 8MB octal PSRAM @80MHz, 64KB/64B data cache). ESP-IDF 5.5.1, bare app,
no WiFi. Raw data: `results/2026-09-23-waveshare-s3cam.jsonl`.

## Headroom with ESP-IDF loaded (`mem`)

| | total | free | largest block |
|---|---|---|---|
| Internal SRAM heap | 316KB | **286KB** | 270KB |
| DMA-capable | 308KB | 278KB | 270KB (same pool) |
| PSRAM | 8.0MB | **8.0MB** | 7.9MB |
| App image / partition | 357KB / 4MB | 3.7MB free | |
| `model` partition | 11MB | | |

Bare IDF costs ~200KB of the 512KB SRAM. WiFi + AFE (and later MicroPython) will
take more; re-measure with them linked.

## Memory tiers, measured

| Tier | Read | Write | Notes |
|---|---|---|---|
| SRAM | 425 MB/s* | 760 MB/s | *scalar C loop, CPU-bound; SIMD loads go higher |
| cache hit (PSRAM ≤ 64KB, hot) | 420 MB/s | 740 MB/s | same as SRAM |
| **PSRAM, streaming** | **88 MB/s** | **49 MB/s** | copy PSRAM→PSRAM 32 MB/s; PSRAM→SRAM 88 MB/s |
| **Flash, mmapped** | **32 MB/s** | — | whole 11MB maps at once; 80% of the QIO bus |
| Flash, `esp_partition_read` | 12.7 MB/s | — | 2.5× slower than mmap; don't use it for weights |
| SD (1-bit) | not measured | | card init timed out (no card inserted?) |

Ratios: PSRAM is **2.75×** flash. Full-tier read time: PSRAM 8MB ≈ 95ms, flash 11MB ≈ 360ms.

## Compute (`fc`, one core, int8, ESP-NN)

1024-wide FC layer. GMAC/s:

| kernel | weights | 1 frame | 4 | 16 | 64 |
|---|---|---|---|---|---|
| plain C | PSRAM | 0.021 | | | |
| PIE, per frame | PSRAM | 0.085 | 0.085 | 0.085 | |
| **PIE, row reuse** | PSRAM | 0.087 | 0.252 | **0.479** | **0.619** |
| **PIE, row reuse** | **flash (mmapped, in place)** | 0.032 | 0.112 | **0.300** | **0.518** |
| PIE, row reuse | SRAM | 0.667 | 0.680 | 0.683 | 0.684 |

(Flash row from a 1024×256 layer, `fc 1024 256 64`; PSRAM/SRAM rows are identical
at that size.)

- **Without weight reuse, PSRAM caps compute at 0.085 GMAC/s**: the weights stream at
  85 MB/s, which is the PSRAM bandwidth. Calling a stock FC kernel once per frame is the
  worst possible structure.
- **Reusing each weight row across N frames** recovers 5.6× at N=16 and 7.1× at N=64
  (90% of the SRAM ceiling). Time fits `weight_bytes/88MB/s + MACs/0.684G` well.
- **Single-core ceiling ~0.68 GMAC/s** with ESP-NN's dot kernel called per row: 18% of
  the 3.84 GMAC/s PIE peak. Call overhead dominates; a kernel that blocks several rows
  and keeps inputs in Q registers should do better. Two cores ≈ 1.3 GMAC/s.
- PIE vs plain C: 4× when PSRAM-bound, 32× from SRAM.

## Dual core (`fc2 1024 256 64`)

Row-reuse kernel, output rows split across two tasks pinned to core 0 / core 1.
Total GMAC/s, speedup vs one core on the same weights:

| weights | 1 frame | 16 frames | 64 frames |
|---|---|---|---|
| SRAM | 1.30 (1.96×) | 1.35 (1.97×) | 1.35 (1.97×) |
| PSRAM | 0.091 (**1.05×**) | 0.90 (1.89×) | **1.18** (1.91×) |
| flash | 0.032 (**1.02×**) | 0.47 (1.58×) | **0.95** (1.84×) |
| flash (core 0) + PSRAM (core 1) | 0.048 | 0.51 | 0.96 |

- **Compute scales ~2× across cores; memory bandwidth doesn't scale at all.** With
  no reuse, a second core adds nothing: flash and PSRAM are one shared bus (MSPI +
  cache), already saturated by one core.
- **Flash and PSRAM traffic is strictly serialized.** Half the weights from each: if
  they take turns, 256KB / (128KB/32 + 128KB/88 MB/s) ≈ 47 MB/s; measured 48 MB/s.
  Streaming experts from flash does *not* come for free alongside PSRAM trunk reads;
  their bus time adds up.
- **With 64-frame reuse the bus stops mattering:** PSRAM 1.18 and flash 0.95 GMAC/s,
  vs a 1.35 SRAM ceiling. At 16 frames, flash-heavy layers lose ~half their second-core
  gain (1.58×).
- Planning rule: **per chunk, time ≈ max(compute on each core, total bus time)** is
  optimistic; a sum is pessimistic. The measurements land between; plan on the
  measured table.

## Consequences for the plan

1. **Chunking isn't optional.** Inference must process 16+ frames per weight pass
   (≥640ms of audio at 25 fps). Per-frame streaming inference from PSRAM is 6–7× slower.
2. **Compute budget (measured, both cores):** ~1.0–1.2 GMAC/s with 64-frame chunks
   (2.56s), ~0.5–0.9 with 16-frame chunks (0.64s), depending on the flash/PSRAM mix.
   At 25 fps that is ~20–45M MAC/frame before headroom for AFE, VAD, WiFi and
   MicroPython. **~5–10M active params is comfortable**, and chunk length is now the
   explicit latency vs throughput knob.
3. **Flash-in-place experts work: confirmed.** The bandwidth model predicted 0.30 /
   ~0.5 GMAC/s at 16 / 64 frames from flash, and it measured **0.300 / 0.518**: 62% /
   84% of PSRAM speed. Routed experts run straight from mmapped flash with no copy-up.
   Copy-up only pays for an expert that stays routed for many chunks: the in-place
   penalty is ~5ms per chunk for a 256KB layer, against an estimated ~13ms one-time
   copy (flash→PSRAM copy speed not yet measured). Default: **in place**; copy up
   only for sticky experts, and only if PSRAM has room.
4. ESP-DL / ESP-NN kernels must be checked for this reuse pattern. A stock per-frame
   FC call would leave 85% of the compute on the table.

## Next benchmarks

- SD, once a card is in.
- ESP-DL 1D conv with `param_copy` true/false: does its conv kernel get this reuse?
