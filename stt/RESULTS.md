# On-device STT results

Board: Waveshare S3 cam (ESP32-S3, 240 MHz, 8MB octal PSRAM, 16MB flash).
Model: NVIDIA Citrinet-256 (9.65M params, CTC), quantized with ESP-PPQ 1.3.11, run by
ESP-DL 3.3.11 with weights read in place from the `model` flash partition (0x410000).
Audio in over WiFi/TCP (`tools/stt.py`), C front end (`main/stt_core.c`, verified
against Python in `export/check_frontend.py`). Raw data: `results/2026-09-23-stt.jsonl`.

## Live voice through the board's own mic (9 clips, 65 words)

| model | on-device WER | RTF (single core) | 2 s command → wait |
|---|---|---|---|
| **w8a16** (int8 weights, int16 activations) | **7.7%** (5 errors) | **2.42** | 4.8 s |
| int8 (CLE) | 9.2% (6 errors) | **1.33** | 2.7 s |
| *(host float, exact length, for reference)* | *4.6%* | | |

- On-device transcripts match the host ESP-PPQ simulation's error pattern: the
  quantization sim is a faithful proxy, so accuracy work can stay on the PC.
- Replayed LibriSpeech clip from the laptop speaker (far field, 13 s): 3 words wrong
  at the end on-device (w8a16), vs 17.9% WER for host float — noise-level differences.
- RTF is flat across clip lengths because the graph is rebuilt at the clip's exact
  length (ESP-DL `input_shapes`, ~540 ms rebuild when the length changes). The fixed
  16 s window cost 33 s for *any* clip (RTF 16.6 on a 2 s clip).
- Front end (log-mel, float FFT): 0.10 × real time. Decode: < 1 ms.

## Where the time goes (w8a16, 16 s window profile)

| op | share | note |
|---|---|---|
| pointwise 1x1 conv (107) | 74% | ~0.15 GMAC/s: compute-bound, weights reused over hundreds of frames |
| residual 1x1 conv (21) | 13% | same kernel |
| ReduceMean (SE, 18) | 4.4% | |
| depthwise conv (107) | 3.8% | |
| decoder conv 256→257 (1) | 3.1% | odd channel count; slow path |

- ESP-DL's `RUNTIME_MODE_MULTI_CORE` changes nothing: it only splits MatMul, not Conv.
- ESP-DL int8 conv ≈ 0.23 GMAC/s effective vs 0.62–0.68 GMAC/s/core for the row-reuse
  PIE kernel in `bench/` (fc/fc2): **~3× kernel headroom per core, plus the second core.**

## Memory

Weights 10.3MB in flash (in place), activations 2.1MB + 0.2MB PSRAM, ~0.4KB internal.
After load: 147KB internal RAM free (before WiFi), 6.1MB PSRAM free.

## What would make it faster / better

1. **QAT for int8**: int8 is 1.85× faster than w8a16 but loses accuracy (sim 5.7–5.9 vs
   3.4 test-clean). Quantization-aware training should close most of that.
2. **Own 1x1-conv kernel on both cores**: the bench kernel suggests ~3× per core, and
   dual core ~2× on top → RTF ≈ 0.2–0.3 (int8). 87% of the time is in that one op.
3. Pad the decoder to 264 (or 256+16) output channels so it takes the fast path.
4. Push-to-talk makes the current RTF usable for short commands already.
