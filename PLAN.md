# esp32-micromodels — PLAN

On-device streaming speech-to-text for the ESP32-S3, built as a **block-routed
mixture-of-experts** laid out across the board's four-tier memory hierarchy: a
resident trunk in SRAM/PSRAM, an expert pool in mmapped flash chosen per utterance
by an audio-embedding router, and a library of expert pools on SD swapped in when
the context changes.

Target host: the Watchtower fleet (MicroPython nodes). Reference board: **Waveshare
S3 cam** (ESP32-S3R8, 8MB octal PSRAM, ES7210 dual-mic, microSD on 1-bit SDMMC
D0=44/CMD=43/CLK=16). Secondary: LilyGo T-Pager (quad PSRAM, ES8311 + mic).

---

## 1. Core idea

For a streaming model, **compute/sec ≈ active params × frames/sec**. At ~25 frames/s
(40ms stride after subsampling) and ~1 GMAC/s sustained, the budget is **~5–10M
active params**. Storage is abundant but slow, so an MoE with total params much larger than active
params is the lever.

Routing is **per block, per utterance** (or per acoustic segment) — never per frame.
LLM-style per-layer/per-token routing would need hundreds of MB/s from storage; the
tiers below PSRAM deliver tens (flash) or single digits (SD).

```
mic → ESP-SR AFE (BSS/NS/VAD) → log-mel (ESP-DSP)                        [SRAM]
    → conv subsampler + router (utterance embedding)                   [SRAM]
    → shared trunk + generalist expert                                  [PSRAM]
    → routed expert blocks, top-1 per block ◄── expert pool (mmapped)   [flash]
    → CTC head (chars / ~256 BPE) → greedy decode → text                [SRAM/PSRAM]

    expert-pool library ──(background refresh when idle, on context change)──► flash
                                                                        [SD]
```

## 2. Memory architecture: one tier per timescale

The board is a four-tier memory hierarchy, each tier ~10× bigger and several times
slower than the one above. The number that matters is **how often a tier can be read
in full** (capacity ÷ bandwidth). That sets how often its contents can change, so
**each tier is assigned to the model timescale that matches its sweep rate.**

Measured on the Waveshare board 2026-09-23 (`bench/RESULTS.md`), except SD:

| Tier | Size (usable) | Effective read | Full read takes | Sweeps | Holds | Changes |
|---|---|---|---|---|---|---|
| **0 SRAM** | **286KB** free after bare IDF (+64KB cache) | ≥425 MB/s (scalar; SIMD higher) | <1 ms | >1000×/s | subsampler, router, activations, current weight tile, audio buffers | **every frame** |
| **1 PSRAM** | 8MB, read/write | **88 MB/s** read, 49 write | ~95 ms | ~10×/s | shared trunk, generalist expert, active experts (if copied up) | **every chunk** |
| **2 Flash** | 11MB `model` partition, read-mostly | **32 MB/s** mmapped | ~360 ms | ~3×/s | the **expert pool** | **every utterance** (routing) |
| **3 SD** | 32GB | ~2–4 MB/s (1-bit bus; *estimate*) | ~3 hours | a few ×/day | the **pool library**, recordings for adaptation, logs | **every context change** |

Compute (measured, int8 FC, one core): **~0.68 GMAC/s** from SRAM; from PSRAM it's
**0.085 GMAC/s with no weight reuse** (bandwidth-bound) and **0.48 / 0.62 GMAC/s**
when each weight row is reused across 16 / 64 frames. Two cores → ~1.1 GMAC/s at best.
Calibration point (ref [1]): an int8 ESP-DL MobileNetV2 at 128×128 (~100M MACs) runs in
94ms on the P4, so ~1 GMAC/s class is typical for these chips.

### Tier rules
1. **Contents change no faster than the tier's sweep rate.** Nothing in the inference
   path waits on a tier slower than its timescale.
2. **Always a fallback one tier up.** The generalist expert (PSRAM) covers a routed
   expert that isn't ready; the current flash pool keeps serving while SD refreshes it.
3. **Tier crossings are where decompression happens.** Experts stored as low-rank
   deltas / int4 / Bloom-filter connectivity (IDEAS.md) expand when copied up a tier.
4. **Compression promotes tiers.** A 4–8× smaller layer can move up a level — a
   pool that needed SD fits in flash, an expert that needed PSRAM fits in SRAM. At
   these ratios, moving up one tier beats almost any kernel optimization.

### The bandwidth walls
1. **PSRAM → core (every chunk).** 5MB of weights × 25 frames/s = 125MB/s, which is more than
   PSRAM delivers. Fix: **chunked inference** — run 8–16 frames per weight pass
   (100–300ms lookahead) with the weight tile in SRAM, so each byte fetched feeds 8–16 MACs.
   **Measured:** per-frame calls run at 0.085 GMAC/s; 16-frame reuse 0.48, 64-frame 0.62.
   Chunking is mandatory, and every kernel we use must reuse weights this way.
2. **Flash → core (every utterance).** A routed expert is either read **in place**
   from mmapped flash (no swap at all, ~half PSRAM speed) or **copied up** into PSRAM
   (~30ms per 1MB, then full PSRAM speed for the rest of the utterance). Measured
   flash is 2.75× slower than PSRAM, and with 16–64-frame reuse that costs only
   ~20–40% of compute, so **in-place looks viable** (to confirm with flash-weights `fc`).
3. **SD → flash (every context change).** Writing flash is slow, wears the chip, and
   **disables the cache while it runs** (stalls both cores). So pool refreshes run
   only while the VAD says it's quiet, rewriting one expert slot at a time.
4. **Flash and PSRAM share one memory bus (MSPI) and one cache.** Flash reads and
   PSRAM reads compete for bandwidth instead of adding up. Measure before relying on overlap.

## 3. Design decisions

- **Toolchain:** ESP-DL (int8/int16) + ESP-PPQ quantization from PyTorch → ONNX
  (opset 18). ESP-SR for the audio front end (AFE) only (not MultiNet — that's a fixed-phrase list, not open vocab).
- **ESP-DL already implements much of the tier plumbing** (ref [1], `dl::Model`):
  - `location`: model in app rodata, **a flash partition (by label)**, or **SD card (by path)**.
  - `param_copy=false`: run weights **in place from mmapped flash** instead of copying
    them to PSRAM — exactly our in-place vs copy-up choice, as a constructor flag.
  - `model_index` / `model_name`: several models packed in one `.espdl`, picked by name.
  - `max_internal_size`: how much internal SRAM the memory planner may use (tier 0 budget).
  So: **start with stock ESP-DL**, split into sub-models (trunk, one per expert block,
  head), and only write custom loaders/kernels where measurement says it's needed.
- **Architecture follows ESP-DL's operator set.** Supported (int8 + int16): 1D/2D Conv
  (groups = 1 or depthwise only), Gemm/MatMul, **GRU, LSTM**, Softmax, Swish/HardSwish,
  ReLU/PReLU, ReduceMean, Sqrt, Div… **No LayerNormalization** (it would be decomposed
  into several quantization-sensitive ops). So the first model family is **QuartzNet /
  Citrinet style**: 1D depthwise-separable conv + BatchNorm (folds into the conv) +
  ReLU, optionally a small GRU for context. Conformers (LayerNorm + attention) come
  later, if at all. Conv ops use NWC layout.
- **Quantize for ESP-DL from day one.** ESP-DL quantization is **symmetric, per-tensor,
  power-of-two scales** (an `exponent`), which is coarse. Lessons from ref [1]: plain ReLU,
  not bounded ReLU6 (~2% on its own there); layerwise equalization + KL
  calibration + bias correction for PTQ; then **QAT recovered a further 2–4%**. For us:
  choose quant-friendly activations up front, train with QAT, and keep sensitive layers
  (first conv, CTC head, router) in **int16** (mixed precision is supported per layer).
- **Experts are deltas, not full blocks** (low-rank / int4 over a resident base
  where possible) → ~10× fewer bytes per swap → finer routing becomes affordable.
  A LoRA delta needs no custom op: `y = Wx + B(Ax)` is just a parallel low-rank
  Conv/Gemm branch plus an Add, which ESP-DL runs natively. The resident base `W`
  lives in PSRAM; the expert is the small `A`,`B` pair in flash.
- **Always-resident generalist expert** as fallback while a routed expert is loading
  or when router confidence is low.
- **Hard top-1 routing, trained for robustness:** randomly substitute wrong/generalist
  experts during training so on-device misroutes degrade gracefully.
- **Experts seeded by clustering:** k-means over trunk utterance embeddings → one
  expert per cluster → joint fine-tune (branch-train-merge style).
- **Silence is free compute — never spend MACs on it.** Three layers:
  1. *Runtime gate:* the ESP-SR voice detector (VAD) wakes the model; idle = front end only. Most of a day
     is silence, so average power/CPU is set by the VAD, not the model.
  2. *Intra-utterance frame dropping:* skip low-energy frames and long pauses inside
     an utterance before the trunk (CTC tolerates this if trained with the same
     dropping). Shortens the sequence the experts see.
  3. *Training data:* trim/normalize silence in the corpus (especially TTS output,
     which has unnaturally clean edges) so the model matches the gated runtime.
- **Tiny output head:** CTC over chars or ~256 BPE, greedy decode. No on-SD n-gram
  beam search (random SD reads are too slow).
- **Runtime placement:** C user module compiled into a custom MicroPython firmware
  (same pattern as our `lvgl_micropython` builds). Inference runs as a FreeRTOS
  task pinned to core 1 — **not** in the asyncio loop (Watchtower `97e0a84` measured
  loop stalls). MicroPython sees only `stt.feed(pcm)` / `stt.poll() -> text`.
- **Flash pool layout: slots are partitions.** Instead of one big `model` partition
  with a custom slot table, the partition table carries N fixed-size **expert
  partitions** (`ex00`…`exNN`, 64KB-aligned), each holding one `.espdl` that ESP-DL
  loads by label (`MODEL_LOCATION_IN_FLASH_PARTITION`, in place or copied up). A pool
  refresh rewrites one partition at a time; a tiny index partition records which
  expert each slot holds and marks it valid only after the write verifies
  (crash-safe, no A/B halving). Custom blob format only if ESP-DL's per-model
  overhead turns out too high.
- **SD library layout:** one directory per context pool (`pools/<context>/`), same
  blob format as flash slots, so a refresh is a plain copy with no re-encoding. SD
  is never read in the inference path.
- **Context manager (SD → flash):** a background task that decides which pool
  belongs in flash (speaker enrollment, language, room/noise profile, time of day,
  or explicit command from Watchtower) and refreshes slots while the VAD is idle.

## 3a. Training data: synthetic speech from TTS

There are many open TTS systems (Piper, Kokoro, MeloTTS, StyleTTS2, XTTS,
F5-TTS, Parler-TTS, …) plus commercial APIs, together covering thousands of voices. We use them
as a data generator:

- **Text side is ours to choose:** general text for coverage, plus a heavy dose of our
  domain (house commands, names, rooms, numbers/times), so rare words the model
  needs get thousands of examples.
- **Diversity over volume:** many engines × voices × speed/pitch/style prompts. A single
  engine teaches the model that engine's vocoder artifacts.
- **Close the TTS→mic gap:** room impulse responses, background noise (MUSAN-style), and
  our actual chain — either replay TTS through a speaker into the ES7210 and record, or
  measure the board's frequency response and apply it as a filter. Replay recordings
  double as a realistic eval set.
- **Always mix in real speech** (LibriSpeech, Common Voice, own captures). Evaluate WER on
  **real** audio only; synthetic-only eval would hide the domain gap.
- **Feeds the MoE directly:** voice/accent/style are controllable, so we can generate
  targeted data per expert cluster and rebalance clusters the router under-serves.
- Pipeline lives in `datagen/`: text sampler → TTS fan-out (resumable job queue, one
  worker per engine) → silence trim → augmentation → sharded manifests.

## 4. Phases

### Phase 0 — Measure the hardware (gates everything)
`bench/` — one ESP-IDF app, one console command per benchmark, results as
`@@R {json}` lines collected by `tools/bench.py` into `results/*.jsonl`.
- [ ] **Headroom with ESP-IDF loaded (`mem`):** free/largest internal SRAM, DMA-capable
      SRAM, PSRAM after boot; app image size vs partition; what's left of flash for
      a model partition. Repeat with WiFi + ESP-SR AFE linked in, since the real firmware
      carries both (and later MicroPython).
- [ ] **Memory bandwidth (`membw`):** read/write/copy for SRAM and PSRAM, octal
      (Waveshare) vs quad (T-Pager), across block sizes vs the 64KB data cache.
- [ ] **Flash mmap bandwidth (`flash`):** sequential read of the `model` partition.
- [ ] **Flash write/erase:** per-64KB-slot erase+write time, and how long the cache
      is disabled (stall seen by a compute task on the other core).
- [ ] **Flash in-place vs copy-up:** `fc` with weights in mmapped flash vs PSRAM,
      plus the flash → PSRAM copy cost → the break-even utterance length.
- [ ] **SD (`sd`):** raw-sector and FAT sequential reads, 1-bit SDMMC at 20/40MHz,
      chunk sizes, DMA into SRAM vs PSRAM.
- [ ] **int8 MAC throughput (`fc`):** ESP-NN PIE kernel vs plain C, weights in SRAM vs
      PSRAM, chunk sizes 1/8/16, direct vs SRAM-tiled weight reuse.
- [ ] SD/flash reads concurrent with compute on the other core (does DMA overlap
      cleanly? how much do flash and PSRAM contend on the shared bus?).
- [ ] **ESP-DL calibration:** a known model (e.g. MobileNetV2-128 from ref [1],
      re-exported for S3) on our board vs the published P4 number → the real S3/P4 ratio.
- [ ] **ESP-DL tier knobs:** the same 1D-conv block exported as `.espdl`, run with
      `param_copy` true vs false (flash in place) and a `max_internal_size` sweep.
- [ ] **ESP-DL per-model overhead:** load/switch time and memory for many small
      sub-models (trunk → expert → head chained) vs one monolithic model.
- [ ] AFE + VAD + log-mel CPU cost with the dual-mic array.

**Output:** `bench/RESULTS.md` → replaces the §2 tier table with measurements, and
fixes the active-param budget, chunk size, flash pool size, and in-place vs copy-up.

### Phase 1 — Dense baseline, desktop
- [ ] TTS data generator (§3a) producing a first ~1000h synthetic set.
- [ ] Small streaming CTC model (QuartzNet / Citrinet class, ESP-DL ops only —
      §3) sized to the Phase 0 budget; train on LibriSpeech + Common Voice + synthetic (+ our own
      mic captures). Ablate: real-only vs real+synthetic, measured on real audio.
- [ ] Train with VAD gating + frame dropping on, matching the runtime.
- [ ] Quantize with ESP-PPQ (PTQ with equalization / KL / bias correction, then
      QAT; int16 for sensitive layers); measure WER float vs PTQ vs QAT on desktop.
- [ ] Record reference audio through the ES7210 so eval matches the real front end.

### Phase 2 — Dense baseline, on device
- [ ] Port to ESP-DL, run in a bare IDF app, streaming from mic. Real-time factor,
      latency, WER on the reference set.
- [ ] This is the bar the MoE has to beat.

### Phase 3 — MoE training
- [ ] Router from trunk embeddings; k-means seeding; per-cluster expert deltas.
- [ ] Routing-robustness training (wrong-expert substitution).
- [ ] Sweep: #experts, expert size, delta rank vs WER, under the constraint that a
      pool fits the ~11MB flash partition. Desktop sim of the tier hierarchy using
      Phase 0 numbers.
- [ ] Context pools: train/cluster several pools (e.g. per speaker group, language,
      acoustic environment) and measure the WER gain of the right pool vs the wrong one
      — does swapping pools on context change pay for itself?

### Phase 4 — MoE runtime on device
- [ ] Expert blob format + exporter; flash slot table; SD pool library format.
- [ ] Routed experts from flash (in place or copied up to PSRAM per Phase 0),
      generalist fallback while an expert isn't ready.
- [ ] Context manager: SD → flash slot refresh, only while the VAD is idle.
- [ ] End-to-end on device: WER, latency, stalls (routing and refresh).

### Phase 5 — Fleet integration
- [ ] C user module in custom MicroPython firmware; `stt` Python API.
- [ ] Watchtower service: transcripts as telemetry/events; server-side Whisper as
      fallback / eval oracle for low-confidence utterances.
- [ ] Experts deployable via OTA/SD update independently of firmware.

## 5. Proposed layout

```
bench/        Phase 0 ESP-IDF benchmark app + RESULTS.md
tools/        host-side scripts (bench runner, …)
results/      raw benchmark results (jsonl), committed
datagen/      TTS synthetic-speech pipeline
train/        PyTorch training, clustering, quantization (uv project)
export/       ONNX → ESP-PPQ → .espdl + expert blob packer
firmware/     ESP-IDF runtime: AFE, mel, inference task, expert cache
mpy_module/   MicroPython C user module wrapper
data/         (gitignored) datasets, mic captures
```

## 6. Open questions / risks

- Is ~5–10M active params enough for usable open-vocabulary WER? Where's the floor?
- What signals a context change cheaply and reliably (speaker embedding drift,
  schedule, Watchtower command)? How many distinct contexts does a household really have?
- Is one ~11MB flash pool enough experts per context, or do we need compression
  (int4 / deltas / Bloom connectivity) to fit a useful pool?
- Flash wear from pool refreshes: fine at a few per day (~100k erase cycles per
  block), but needs a cap so a flapping context signal can't thrash it.
- ESP-DL has the ops for a conv/GRU CTC model (§3), but **streaming state** (cached
  conv context between chunks, GRU hidden state carried across calls) — does it
  handle this, or do we manage state tensors ourselves around `model->run()`?
- Custom kernels (Bloom connectivity, fused delta+base) mean hand-written PIE
  assembly: no compiler auto-vectorization, only 8 × 128-bit Q registers, and
  S3 (Xtensa `ee.*`, 160-bit QACC halves) vs P4 (RISC-V `esp.*`, 256-bit QACC) kernels
  **do not port** — each chip needs its own. The S3's 8-bit MAC lanes accumulate in
  ~20 bits, so long dot products must go through the 40-bit ACCX or be split (ESP-NN's
  kernels already handle this).
- Can the router decide reliably from the first ~0.5s of an utterance?
- Is PSRAM contention with the camera (on camera nodes) a problem while STT runs?
- Would an ESP32-P4 target be worth a parallel track? Ref [1] shows ~1 GMAC/s class
  int8 inference there with the same ESP-DL model files (re-exported per target), so
  a model developed for the S3 moves up cleanly, but custom PIE kernels would not.

## 7. References

1. **esp32-p4-vehicle-classifier** — https://github.com/boumedinebillal/esp32-p4-vehicle-classifier
   INT8 MobileNetV2 (3.5M params, 2.6MB) via ESP-DL/ESP-PPQ on the P4: 96px 70ms,
   128px 118ms total (94ms model, 80%), 256px 459ms. PTQ (equalization, KL, bias
   correction) → QAT; ReLU6→ReLU. Loads the model from app rodata with ESP-DL
   `param_copy` default (copy to PSRAM). Ships an S3 sdkconfig identical to ours
   (octal PSRAM 80MHz, 64KB/64B dcache) but reports no S3 numbers. Useful as a
   training/export pipeline template.
2. **Espressif, "Introduction to the ESP32-P4/S3 PIE"** —
   https://developer.espressif.com/blog/2024/12/pie-introduction/
   8 × 128-bit Q registers (16×int8 / 8×int16 / 4×int32 lanes); S3 QACC 2×160-bit,
   P4 2×256-bit; 40-bit ACCX; loads/stores want 128-bit alignment (unaligned
   supported via extra instructions); written as inline asm or `.S`. P4-only benchmarks
   (memcpy 74% faster than libc; int16 vector add 94% faster than C). No S3 numbers,
   no comments section.
