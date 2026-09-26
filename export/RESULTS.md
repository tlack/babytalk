# Citrinet-256 on the host: float port → static ONNX → ESP-PPQ (esp32s3)

> The ESP-PPQ / ESP-DL sections below are the project's first pipeline (2026-09-23), kept as
> history: the board now runs MMRT at each clip's exact length (no static 16 s window), and
> int8/int4 rather than w8a16. The last section, "4-bit weights", reflects the current models.

Question: what WER does NVIDIA Citrinet-256 (`stt_en_citrinet_256_ls`, 9.8M params, CTC,
WordPiece vocab 256 + blank) get on LibriSpeech test-clean (a) as our plain-PyTorch port,
(b) quantized the way ESP-DL would run it on an ESP32-S3, simulated with ESP-PPQ 1.3.11?
Greedy CTC, no LM. Published: **3.78** test-clean.

## Results

| # | model / pipeline | utts | WER % |
|---|---|---|---|
| 1 | **Float, our port** (exact lengths, NeMo masking) | 2620 (all) | **3.764** |
| 1b | same, BatchNorm folded | 2620 | 3.764 |
| 2 | Static 1600-frame window, **zero-padded**, long clips cut in equal chunks | 2620 | 4.337 (≤16 s: 4.07, >16 s: 5.42) |
| 3 | **Static window, tiled fill + overlapped windows** (the deployable pipeline) | 2620 | **3.728** (≤16 s: 3.76, >16 s: 3.59) |
| 3b | same through the exported ONNX in onnxruntime (dead SEs folded) | 2620 | 3.728 |
| 4 | ESP-PPQ **int8** (w8a8), esp32s3 | 655 | 5.917 |
| 4b | int8 + our dw→pw cross-layer equalization (CLE) | 655 | 5.685 |
| 5 | int8 + int16 for block-0 convs & CTC decoder (mixed) | 655 | 5.819 (CLE: 5.84 on 328) |
| 6 | **ESP-PPQ w8a16** (int8 weights, int16 activations) + CLE | 655 | **3.662** |
| 7 | ESP-PPQ int16 (w16a16) | 655 | 3.423 |
| — | float, same static pipeline, same 655 utts (reference for rows 4–7) | 655 | 3.438 |

Rows 4–7 use the eval subset = every 4th test-clean utterance (655; 1.3 h of audio);
calibration = 128 **dev-clean** windows (never overlaps eval). On that subset the float
static pipeline scores 3.438, so the deltas are: int8 **+2.25–2.5 abs (≈+70% rel)**,
mixed int16 first/last layers +2.4, **w8a16 +0.22**, int16 ±0.

Other experiments (328-utt subset, every 8th; float 3.738 there; int8 6.15, CLE int8 5.76):
int16 for all 107 depthwise convs 5.70; w8a16 only in blocks 12–22 + decoder 4.99; w8a16
only in blocks 0–11 6.15. PPQ's own layerwise equalization and bias correction were each
still running after 40+ min on 24 cores and were abandoned (not evaluated).

### What the numbers say
- **The port is right**: 3.764 vs 3.78 published, on all 2620 utterances.
- **Static shape costs nothing** if the window is filled well. Zero-padding hurts because
  the global SE (squeeze-excite) mean is taken over the whole window, padding included.
  Filling the window with repeats of the clip separated by 16 zero frames makes the SE mean
  ≈ the clip mean: no graph change, WER back to float. Clips > 16 s: half-overlapped 16 s
  windows (hop 800 frames), each output frame taken from the window whose centre is nearest.
  (A static graph with explicit length masks also works — 3.79 on ≤16 s with SE-only
  masking — but needs extra inputs; the tiling trick is simpler.)
- **int8 PTQ loses ~2.3 WER, and it's the activations, not the weights.** w8a16 keeps
  the same int8 weights (same ~10 MB) and lands within 0.2 of float. Noise is spread over
  the whole depth (graphwise SNR: noise/signal grows steadily to 17% at the logits; no single
  bad layer), so int16 on a few "sensitive" layers doesn't help. Reason: on the **S3, ESP-DL
  weights are per-tensor power-of-2** (per-channel is P4-only in ESP-PPQ's quantizer), and
  activations are per-tensor pow2 too. Getting int8 activations to work needs QAT.
- **5 of the 23 SE gates are dead** in this checkpoint (blocks 0, 10, 13, 14, 15: weights
  ≤ 8e-3, fc1 output exactly 0 on dev-clean → gate ≡ sigmoid(0) = 0.5). Left in, they
  broke ESP-DL's int8 exponent rule (`out_exp ≥ in_exp + w_exp` violated on block 10's
  fc1). We fold the 0.5 into the preceding conv and drop them; WER unchanged.

## Static model / board interface

| | |
|---|---|
| ONNX (float, opset 18) | `data/models/citrinet256_static1600.onnx` (38.7 MB); CLE variant `…_cle.onnx` (numerically equivalent) |
| ONNX input / output | `feats` [1, 80, 1600] NCW float → `logits` [1, 257, 200] (pre-softmax; blank = 256) |
| **.espdl input** | `feats` **[1, 1600, 80] NWC** (time-major, 80 mels per frame) |
| **.espdl output** | `logits` [1, 200, 257] |
| int8 `.espdl` | `data/models/citrinet256_s3_cle_int8.espdl` (10.07 MB): input **INT8, exponent −5** (x_q = round(x·32), clamp ±127 → covers ±3.97); output INT8 exp −2 |
| w8a16 `.espdl` (**recommended**) | `data/models/citrinet256_s3_cle_w8a16.espdl` (10.86 MB): input **INT16, exponent −11** (x_q = round(x·2048), covers ±16); output INT16 exp −9 |
| also exported | `citrinet256_s3_int8.espdl` (no CLE; in exp −5), `…_int8_mixed16.espdl` (in INT16 exp −12), `…_cle_int8_mixed16.espdl`, `…_int16.espdl` (19.8 MB; in INT16 exp −11) |

Exponents are chosen by calibration, so re-read them after re-quantizing (they're in the
`.info` sidecar next to each `.espdl`, and ESP-DL exposes them on the input tensor at runtime).
Decode: argmax per frame over 257, collapse repeats, drop blank (256) and `[PAD]/[UNK]/[CLS]/[SEP]/[MASK]`,
WordPiece join (`##x` continues a word), and glue `'` to both neighbours
(the tokenizer split "don't" into `don ' t`). Decode only the first `ceil(valid_frames/8)`
output frames.

**Features the board must produce** (NeMo `AudioToMelSpectrogramPreprocessor`, as `citrinet.MelFeatures`):
16 kHz float audio in [−1, 1]; pre-emphasis 0.97 (`y[0]=x[0]`); STFT n_fft 512, win 400
**symmetric** Hann (`periodic=False`), hop 160, centred with zero padding of 256 each side;
power |X|²; 80-band slaney mel, 0–8 kHz, slaney norm (== `librosa.filters.mel(16000, 512, 80)`,
also stored in the checkpoint as `preprocessor.featurizer.fb`); `ln(mel + 2^-24)`;
frames = samples//160 + 1; **per-feature normalization over the clip's valid frames**:
(x − mean) / (std_unbiased + 1e-5); dither 0. Then fill the 1600-frame window by tiling
[clip, 16 zero frames, clip, 16 zeros, …]. STFT pad mode (constant vs reflect) is
immaterial (≤0.003 WER).

## Ops in the exported graph vs ESP-DL (esp32s3)

ONNX (after dead-SE folding): Conv ×272 (107 depthwise k=5…41, 107 pointwise, 36 SE-FC 1×1,
21 residual 1×1 of which 3 strided, 1 decoder), Relu ×125, ReduceMean ×18, Sigmoid ×18,
Mul ×18, Add ×21, plus 18 `Constant` axes (folded by the simplifier). **Everything is on
ESP-DL's supported list** (`operator_support_state.md`): Conv 1d with groups = 1 or =
C_in ✔ int8/int16/w8a16; Relu, Add, Mul, ReduceMean, Sigmoid ✔ int8/int16. In the
exported `.espdl`, Relu is fused into the preceding Conv (103 of 125), Sigmoid becomes a
LUT, SE's global mean + FC + sigmoid + mul map directly (ReduceMean over time → two 1×1
Conv on [1,1,C] → Sigmoid LUT → broadcast Mul). In w8a16 mode, depthwise convs are
exported as full int16 (S16) and pointwise as W8A16. Mixed-precision boundaries insert
`RequantizeLinear`, which ESP-DL implements (`dl_module_requantize_linear.hpp`) though it's
not in the op table. LogSoftmax was dropped (argmax doesn't need it). No unsupported ops.

## Size and compute

| | |
|---|---|
| Params | 9.77M in the checkpoint (incl. BN); 9.65M after BN fold + dead-SE removal |
| MACs | 4.34 G per 16 s window = **271 M MAC per second of audio** (94% pointwise 1×1, 6% depthwise) |
| Element-wise ops | 20.5 M per window |
| .espdl size | int8 10.07 MB, w8a16 10.86 MB, int16 19.84 MB |
| Activations | largest tensor 256×1600 = 400 KB int8 / 800 KB int16; naive liveness peak 1.2 MB int8 / 2.4 MB int16 (our estimate — ESP-PPQ/ESP-DL tooling reports none on the host; the on-device planner decides) → PSRAM, not SRAM |

Against the bench numbers (`bench/RESULTS.md`: ~1.0–1.2 GMAC/s int8 both cores with 64-frame
reuse) that's roughly **0.25–0.3× real time for int8** if ESP-DL's conv kernel reaches the
bench's reuse rate — unmeasured. w8a16 will be slower (16-bit lanes; not measured). The
weights (~10 MB) exceed the 8 MB PSRAM, so they'd run in place from the 11 MB flash
partition (`param_copy=false`); per window each weight byte is reused across 1600/800/400/200
frames, so the bus shouldn't be the limit.

## Surprises / notes
- The tokenizer is **WordPiece (BERT) `vocab.txt`**, not SentencePiece. Its punctuation split
  makes `'` a free-standing token; joining tokens with spaces gives 6.47 WER instead of 3.76.
- Five dead SE blocks in a released NVIDIA checkpoint; they also trip ESP-DL int8 export.
- ESP-PPQ overwrites its input ONNX with the onnx-simplified version, so `quantize.py`
  quantizes a copy in `data/models/ppq/`.
- ESP-PPQ installs fine on Python 3.12 with CPU torch (uses `onnxsim-prebuilt`); the whole
  env is ~1.5 GB.

## Reproduce (from `export/`)
```
uv run evalwer.py float                                   # row 1
uv run check_fold.py && uv run evalwer.py float --fold    # BN fold check, row 1b
uv run evalwer.py fixed --fold                            # row 2
uv run evalwer.py fixed --fold --fill tile16 --long overlap   # row 3
uv run export_onnx.py [--cle]                             # ONNX + ORT check
uv run evalwer.py onnx --fill tile16 --long overlap       # row 3b
uv run quantize.py                                        # row 4 (+ --mixed, --int16, --qtype w8a16)
uv run quantize.py --src ../data/models/citrinet256_static1600_cle.onnx --tag cle_int8          # 4b
uv run quantize.py --src ../data/models/citrinet256_static1600_cle.onnx --qtype w8a16 --tag cle_w8a16   # row 6
uv run stats.py                                           # params / MACs / activations
```
Per-utterance hypotheses: `data/results/*.tsv` (uid, duration, ref, hyp); summaries in
`data/results/summary.jsonl` and `quant_summary.jsonl`; per-op exponents in
`data/results/exponents_<tag>.tsv`. Data: LibriSpeech test-clean + dev-clean in `data/librispeech/`.

## Board mic recordings (data/recordings, 2026-09-23)

(ESP-DL era: the static-window and w8a16 conclusions here were overtaken by MMRT.)

9 live-voice clips of the user speaking to the Waveshare board's ES7210 mic 1
(16 kHz, gain 14, normal speaking range; 65 words total) plus one LibriSpeech clip
replayed from the laptop speaker across the room. `uv run recordings.py` (float) and
`uv run quantize.py … --eval-set recordings` (int8 sims). One word = 1.5% WER here, so
treat differences of a word or two as noise.

| pipeline | live voice WER | errors |
|---|---|---|
| float, exact lengths | **4.62%** | hello world**s**; claude opus → cloopus |
| float, static 1600-frame window (tiled fill) | 6.15% | + hello → hellow |
| **w8a16 sim (esp32s3)** | **7.69%** | + in → and |
| int8 sim (esp32s3, CLE) | 10.77% | + town → towning, union → uniony, cloopus → claodopus a |

- Clean source vs board mic is a small gap at speaking range: most live clips are
  word-perfect. The misses are a proper noun LibriSpeech never saw ("Claude Opus")
  and short-clip edge effects.
- Laptop-speaker replay across the room is the hard case: 17.9% float vs 0% on the
  same clip clean (−43 dBFS speech, ~11 dB over the room floor).
- Very short clips (2 s) get tiled ~8× to fill the 16 s window; "hellow" appears only
  in the static pipeline, so the tiling fill is worth revisiting for short push-to-talk
  commands (e.g. a shorter static window, or fill with silence + masked SE).
- int8's extra errors on real audio match its test-clean loss (5.7–5.9 vs 3.4); w8a16
  stays close to float. **The board should run w8a16** until QAT closes the int8 gap.

## 4-bit weights (CB4 codebooks, GPTQ) — 2026-09-24

WER through the host C executor (bit-exact with the board; clip-length input like the
board, no window tiling, so int8 here differs slightly from the PPQ rows above).
`int4_eval.py` / `int4_gptq.py`, 128 dev-clean calibration clips.

| weights | test-clean (655) | recordings | your voice | image |
|---|---|---|---|---|
| int8 | 6.28 | 12.40 | 12.31 | 9.78 MB |
| 4-bit GPTQ (act-order + bias corr.) | 8.57 | 10.74 | 9.23 | 5.90 MB |
| same, first 1x1 + decoder int8 | **8.21** | 10.74 | 9.23 | ~6.1 MB |

Plain per-channel 4-bit k-means without GPTQ: 96% (27-utt slice). On the slice,
act-order is worth ~2.2 WER, bias correction ~0.2. The recordings set is small: int4
scoring better than int8 there is within its noise.
