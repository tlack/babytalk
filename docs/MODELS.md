# Models: using ours, building your own, and other languages

BabyTalk runs two neural models on the board, and they are replaced in very different ways:

| | Speech to text | Text to speech |
|---|---|---|
| Model | NVIDIA Citrinet-256 (CTC), converted to BabyTalk's `.mmrt` format | sanoTTS `en_us_e12nano` voice (294k parameters) |
| Where it lives | its own flash partition (`model`), read in place | compiled into the firmware |
| Replace it by | flashing a different `.mmrt` file | rebuilding the firmware |
| Language | English (a new language: see [section 4](#4-another-language)) | English |

This page goes from easy to hard: use our models (1), use your own fine-tuned variant (2),
change wake phrases (3), another language (4), a different speech model architecture (5),
and a different voice (6). Each says what it takes and what we have and haven't verified.

## 1. Our speech models

Both are in `models/` (details, SHA-256 and licence in [`models/README.md`](../models/README.md)):

| file | size | word error rate (LibriSpeech test-clean) | fits |
|---|---|---|---|
| `citrinet256_int4.mmrt` | 5.99 MB | 8.2% | the standard 6 MB `model` partition |
| `citrinet256_int8.mmrt` | 9.78 MB | 6.3% | only a 10 MB partition (MicroPython's int8 layout) |

The original full-precision model scores 3.8%; the int8 and int4 versions lose accuracy to
quantization (section 2 has what we plan to do about it).

**Flashing one** (esptool, from the repo root):

| firmware | `model` partition | int4 | int8 |
|---|---|---|---|
| MicroPython, standard layout | 0x410000, 6 MB | yes | no |
| MicroPython, int8 layout (`partitions-stt-int8.csv`) | 0x410000, 10 MB | yes | yes |
| AtomVM (`atomvm/partitions-babytalk.csv`) | 0x490000, 6 MB | yes | with a 10 MB partition (see below) |

For the int8 model on AtomVM, grow the model partition to 10 MB in
`atomvm/partitions-babytalk.csv` (model `0x490000` size `0xA00000`, then `main.avm` at
`0xE90000` size `0x170000`, 1.4 MB for your app) and rebuild; apps then flash to `0xE90000`.
Not yet tried on the board.

```bash
esptool.py --chip esp32s3 write_flash 0x410000 models/citrinet256_int4.mmrt   # MicroPython
esptool.py --chip esp32s3 write_flash 0x490000 models/citrinet256_int4.mmrt   # AtomVM
```

Or, from MicroPython, install a model from the SD card or the board's filesystem without a
computer: `mpy/examples/model_tools.py` (`model_tools.install("/sd/citrinet256_int4.mmrt")`
copies, verifies and resets). The format (int8 or 4-bit, per layer) is recorded in the file,
so the firmware needs no setting either way.

## 2. Your own fine-tuned variant

This is the path for making the model better at *your* conditions (your voice, your room,
your vocabulary, background noise), in the same language. The tokenizer and audio features
stay the same, so **the result is just another `.mmrt` file: no firmware change**.

1. **Record** on the board in your conditions: `field/capture.py` (a session plays a
   background sound, shows sentences, records you and synthetic voices). Sentences come from
   `field/prompts.py` (audiobooks, news, Wikipedia, Hacker News, your own
   `field/terms.txt`). A fifth of the sentences are held out for testing.
2. **Fine-tune** on the PC: `train/make_tts.py` (synthetic speech of the training sentences),
   then `train/finetune.py` (~15 minutes on a laptop GPU, ~1.5 days on its 24-thread CPU with
   `--device cpu`; Apple's GPU: `--device mps`, untested; mixes audiobooks, synthetic speech
   and your recordings, with recorded noise added). Blending with the original weights trades
   noisy-room accuracy against clean-speech accuracy. The result is a PyTorch checkpoint.
3. **Score** it before converting: `export/field_eval.py --float-ckpt <checkpoint>`.
4. **Convert** to the board's format (from `export/`). `CITRINET_CKPT` points the scripts at
   your checkpoint; their files then go in an `export/` directory beside it:

   ```bash
   export CITRINET_CKPT=$PWD/../data/train/runs/<run>/model.pt
   uv run export_onnx.py --cle && uv run mmrt_quant.py && uv run mmrt_export.py    # int8
   uv run mmrt_check_model.py                                    # C runtime bit-exact with the graph
   uv run int4_gptq.py --keep-io                                 # 4-bit weights (~20 min)
   D=../data/train/runs/<run>/export/mmrt
   uv run mmrt_cb4.py $D/citrinet256_cb4_gptq16_io.mmrt -o $D/citrinet256_int4.mmrt
   ```

   Without `CITRINET_CKPT` the same commands rebuild the stock models from NVIDIA's weights.
5. **Check** the converted file on the PC (`field_eval.py --mmrt my_int4.mmrt` runs the same
   C code as the board, bit for bit) and on the board (`export/mmrt_check_device.py`), then
   flash it as in section 1.

**Quantization loss.** Today the 4-bit model gives up ~4.4 points of word error rate against
full precision on clean audiobooks (8.2% vs 3.8%). Quantization-aware training -- fine-tuning
with the board's 8-bit arithmetic and 4-bit codebooks simulated -- is the standard way to win
much of that back. It is planned, not done, and needs a faithful simulation of MMRT's rounding.

**Licence:** a model fine-tuned from NVIDIA's weights is still derived from them (CC-BY-4.0):
keep the attribution in `models/README.md` when you share it.

## 3. Wake phrases

No training and no new model: any phrase works, scored against the speech model's output.
Give it a few spellings of how the model tends to hear the phrase:

- `export/kws.py --phrase "hey jarvis" --tokens` shows how a phrase breaks into the model's
  word pieces, and scores spellings against recordings to pick a threshold
  (`tools/wake_phrases/*.json` has a tested one). `tools/wake.py "hey jarvis" --enroll 3`
  (with the `stt/` test firmware) has you say it three times on the board and saves the
  spellings the model heard.
- MicroPython: `stt.phrase([...])`; AtomVM: `babytalk:phrase([...])` or
  `babytalk_listener:wake_on/2`.
- `atomvm/apps/wakeword_demo` enrolls a phrase by voice on the board: it stores the model's
  own transcription of how you said it, which is the spelling it will match best.

Short phrases wake more easily by accident; the default threshold (-21) was tuned for "wake
up tomato face".

## 4. Another language

*"This is cool, but how do I use Spanish?"* -- the honest answer is that it's a project, not
a download.

**There is no small off-the-shelf model.** NVIDIA's Citrinet-256 exists only in English. The
non-English Citrinets on Hugging Face are the 512 and 1024 sizes: `nvidia/stt_es_citrinet_512`
(Spanish, 1024-token vocabulary), NeonGecko's 512s for Spanish, German, French, Italian,
Portuguese, Catalan, Dutch and Ukrainian, and 1024s for Chinese. A Citrinet-512 has about 36M
parameters: ~18 MB even at 4 bits, more than the board's whole 16 MB of flash (and the weights
must be read in place from flash; 8 MB of PSRAM can't hold them either).

**The realistic route is to train a Spanish Citrinet-256**, the way NVIDIA built their
Spanish 512: keep the English encoder, give it a new output layer for a Spanish tokenizer, and
fine-tune on Spanish speech (Mozilla Common Voice, Multilingual LibriSpeech). `train/finetune.py`
is the starting point; expect hours to days of GPU time and your own evaluation. Distilling
the Spanish 512 into a 256 is another option. We haven't done either.

**Then the firmware needs changes**, because the tokenizer is compiled in:

- **Vocabulary and features**: `export/gen_c_tables.py` regenerates
  `stt/main/citrinet_tables.h` (the mel filterbank and the token list) from the new model;
  rebuild the firmware.
- **Token convention**: our decoder (`stt_ctc_greedy` in `stt/main/stt_core.c`) and the wake
  phrase scorer (`stt/main/kws.c`) assume BERT-style WordPiece, where `##` marks a word
  continuation. NVIDIA's Spanish models use SentencePiece, where `▁` marks a word start. Both
  need a small change to support it.
- **Output size**: the exporter pads the 257 outputs (256 tokens + blank) to 272 for the SIMD
  kernel. A 1024-token vocabulary pads to 1040: the same mechanism, not yet tested at that size.
- **Text**: accents and `ñ` come through as UTF-8; check what your application does with them.

**Size budget.** The model partition is 6 MB (6,291,456 bytes), and today's 4-bit model
already uses 5,991,232 of it. The output layer maps 640 channels to the vocabulary: 174 KB at
8 bits for 256 tokens, 666 KB for 1024. So a 1024-token model doesn't fit as is: quantize the
output layer to 4 bits too (at some cost in accuracy), use a smaller vocabulary, or grow the
partition (both firmwares' partition tables have room to trade against the filesystem).

Speaking Spanish is a separate problem: see section 6.

## 5. A different speech model architecture

MMRT runs what Citrinet needs and nothing more: 1-D depthwise and 1x1 convolutions, a mean
over time, lookup tables (for activations like sigmoid), elementwise multiply and add, on
time-major int8 tensors with power-of-two scales. `export/mmrt_export.py` builds the model file
from an ESP-PPQ-quantized graph made of those ops. Another 1-D convolutional CTC model built
from the same ops (QuartzNet-style, for example) *should* go through the same chain; we
haven't tried one. Anything with attention, recurrent layers (LSTM/GRU), 2-D convolutions or
an FFT inside the network needs new MMRT kernels first.

## 6. A different voice, or another language for speech output

The voice is compiled into the firmware from a pinned sanoTTS checkout
([`components/sanotts/`](../components/sanotts/)), so changing it means changing what the
build compiles in. Things to know:

- **Memory**: the voice runs in the shared 84 KB internal-RAM block (its "arena"), plus ~104 KB
  of other buffers (PSRAM in the AtomVM build). A bigger voice may need a bigger arena, which
  may not fit next to WiFi.
- **Other English voices**: sanoTTS publishes a microcontroller build of a larger English voice
  (`mcu-kristin-745k-q8`, in its voices-v1 release). Swapping it in means changing the model
  data file the build compiles and checking its arena size. Untested here.
- **Other languages**: sanoTTS has Spanish, German, French, Italian, Portuguese and more
  (511k-1.56M parameters), but as browser voices (fp16), not published as microcontroller
  builds, and they get their pronunciation from espeak-ng. Our board voice uses a small
  English-only pronunciation dictionary instead. For Spanish on the board you'd need an int8
  microcontroller build of a Spanish voice (sanoTTS's tooling, not ours) and a Spanish
  pronunciation step: espeak-ng (GPL-3.0, and ~62 KB of internal RAM, too much next to
  AtomVM and WiFi) or a small rule-based Spanish G2P, which is realistic because Spanish
  spelling is regular.
- **Licensing**: read [`components/sanotts/LICENSES.md`](../components/sanotts/LICENSES.md)
  before distributing firmware with a voice in it.
- **Speed and pace**: `length_scale` (AtomVM `say/2`, `tts_set_length_scale()` in C) slows or
  speeds any voice without retraining.

## Checklist for any new model

- **Fits**: the speech model in its partition (6 MB, or 10 MB with the int8 layout), the
  voice's arena in the 84 KB shared block.
- **Same audio**: 16 kHz mono in, 80-bin log-mel features as `citrinet_tables.h` describes.
- **Checked**: word error rate on held-out audio (`export/field_eval.py`) and the board
  bit-exact against the PC (`export/mmrt_check_device.py`).
- **Licensed**: know the terms of the weights you started from and of any voice you compile
  in; keep the attribution they ask for.
