# esp32-micromodels — PLAN

On-device streaming speech-to-text for the ESP32-S3, built as a **block-routed
mixture-of-experts**: a small resident trunk plus expert blocks streamed off the SD
card, chosen per utterance by an audio-embedding router.

Target host: the Watchtower fleet (MicroPython nodes). Reference board: **Waveshare
S3 cam** (ESP32-S3R8, 8MB octal PSRAM, ES7210 dual-mic, microSD on 1-bit SDMMC
D0=44/CMD=43/CLK=16). Secondary: LilyGo T-Pager (quad PSRAM, ES8311 + mic).

---

## 1. Core idea

For a streaming model, **compute/sec ≈ active params × frames/sec**. At ~25 frames/s
(40ms stride after subsampling) and ~1 GMAC/s sustained, the budget is **~5–10M
active params**. SD storage is effectively free, so an MoE with total params much larger than active
params is the lever.

Routing is **per block, per utterance** (or per acoustic segment) — never per frame.
LLM-style per-layer/per-token routing would need hundreds of MB/s from SD; we have
single-digit MB/s.

```
mic → ESP-SR AFE (BSS/NS/VAD) → log-mel (ESP-DSP)
    → [resident] conv subsampler + shared trunk  ──► router (utterance embedding)
    → [swapped]  expert blocks k∈top-1, from PSRAM cache ◄── SD prefetch (DMA)
    → [resident] CTC head (chars / ~256 BPE) → greedy decode → text
```

## 2. Hardware budget (estimates — Phase 0 replaces these with measurements)

| Resource | Estimate | Constrains |
|---|---|---|
| PIE SIMD peak | ~7 GMAC/s int8 (2 cores) | ceiling only |
| Sustained int8 | ~0.5–1.5 GMAC/s | active params |
| Internal SRAM | 512KB | hot weights: subsampler, router, activations |
| Octal PSRAM | 8MB, ~40–80 MB/s effective | resident trunk + expert cache; **weight reuse** |
| SD 1-bit SDMMC | ~2–4 MB/s | expert swap latency → routing granularity |

### The two bandwidth walls
1. **SD → PSRAM.** A 1MB expert takes ~300–500ms to load. So: route per utterance,
   keep an LRU expert cache in PSRAM, prefetch by DMA while the cores compute.
2. **PSRAM → core.** 5MB of weights × 25 frames/s = 125MB/s, which is more than PSRAM delivers.
   Fix: **chunked inference** — run 8–16 frames per weight pass (100–300ms lookahead),
   so each byte fetched feeds 8–16 MACs.

## 3. Design decisions

- **Toolchain:** ESP-DL (int8/int16) + ESP-PPQ quantization from PyTorch → ONNX.
  ESP-SR for the audio front end (AFE) only (not MultiNet — that's a fixed-phrase list, not open vocab).
- **Experts are deltas, not full blocks** (low-rank / int4 over a resident base
  where possible) → ~10× fewer bytes per swap → finer routing becomes affordable.
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
- **SD file format:** one flat, aligned blob per expert (header + int8 tensors in
  ESP-DL layout) so a load is a single sequential DMA read, no parsing.
- **Flash is a second expert store.** 16MB flash minus the app leaves ~11MB for a
  `model` data partition, readable through the cache via `esp_partition_mmap`.
  If mmapped flash beats 1-bit SD (Phase 0 decides), the hot experts live in flash
  and SD holds the long tail.

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
- [ ] **SD (`sd`):** raw-sector and FAT sequential reads, 1-bit SDMMC at 20/40MHz,
      chunk sizes, DMA into SRAM vs PSRAM.
- [ ] **int8 MAC throughput (`fc`):** ESP-NN PIE kernel vs plain C, weights in SRAM vs
      PSRAM, chunk sizes 1/8/16, direct vs SRAM-tiled weight reuse.
- [ ] SD/flash reads concurrent with compute on the other core (does DMA overlap cleanly?).
- [ ] ESP-DL conv block from a real exported `.espdl` model.
- [ ] AFE + VAD + log-mel CPU cost with the dual-mic array.

**Output:** `bench/RESULTS.md` → fixes the active-param budget, chunk size, the
minimum routing interval, and SD-vs-flash for expert storage.

### Phase 1 — Dense baseline, desktop
- [ ] TTS data generator (§3a) producing a first ~1000h synthetic set.
- [ ] Small streaming CTC model (QuartzNet / small conformer class) sized to the
      Phase 0 budget; train on LibriSpeech + Common Voice + synthetic (+ our own
      mic captures). Ablate: real-only vs real+synthetic, measured on real audio.
- [ ] Train with VAD gating + frame dropping on, matching the runtime.
- [ ] Quantize with ESP-PPQ; measure WER float vs int8 on desktop.
- [ ] Record reference audio through the ES7210 so eval matches the real front end.

### Phase 2 — Dense baseline, on device
- [ ] Port to ESP-DL, run in a bare IDF app, streaming from mic. Real-time factor,
      latency, WER on the reference set.
- [ ] This is the bar the MoE has to beat.

### Phase 3 — MoE training
- [ ] Router from trunk embeddings; k-means seeding; per-cluster expert deltas.
- [ ] Routing-robustness training (wrong-expert substitution).
- [ ] Sweep: #experts, expert size, delta rank vs WER. Desktop sim of swap latency
      using Phase 0 SD numbers.

### Phase 4 — MoE runtime on device
- [ ] Expert blob format + exporter.
- [ ] PSRAM LRU cache, DMA prefetch, generalist fallback during loads.
- [ ] End-to-end on device: WER, latency, swap stalls.

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
- Does 1-bit SD (Waveshare wiring) make swaps too slow? Could a different board use 4-bit SDMMC?
- Does ESP-DL support the ops we need for streaming (stateful/cached convs,
  attention if we go conformer), or do we write custom PIE kernels?
- Can the router decide reliably from the first ~0.5s of an utterance?
- Is PSRAM contention with the camera (on camera nodes) a problem while STT runs?
- Would an ESP32-P4 target (faster core, more memory bandwidth) be worth a parallel track?
