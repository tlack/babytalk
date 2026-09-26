# BabyTalk

![BabyTalk: small AI models for speech to text and text to speech on ESP32 microcontrollers](docs/babytalk-splash.png)

**BabyTalk** makes an ESP32-S3 that **understands what you say and talks back, with no
cloud and no internet**, driven from MicroPython:

```python
import stt, tts
text = stt.transcribe(pcm)                 # 16 kHz audio -> "what's the weather like today"
audio = tts.say("You said: " + text)       # text -> 24 kHz audio for the speaker
```

It also runs from **Erlang and Elixir** on [AtomVM](https://github.com/atomvm/AtomVM): see
[BabyTalk for AtomVM](#babytalk-for-atomvm-erlang--elixir).

Made by Thomas Lackner and Claude Opus 5.5 (Anthropic). MIT licensed; the speech models
belong to their authors (see [Credits and licenses](#credits-and-licenses)).

## Is this for you?

Most "voice on ESP32" projects do one of two things:

- **send the audio to a cloud service** (Whisper API, Google, wit.ai) and wait for text, or
- **recognize a fixed list of commands or wake words** that were trained in advance
  (e.g. ESP-SR's MultiNet, Edge Impulse / TensorFlow Lite Micro keyword spotting).

BabyTalk is different: a **general-purpose English speech recognizer** that runs
entirely on the chip. It writes down *any* sentence, with no training and no command list.
A wake phrase is any phrase you type ("wake up, tomato face"), also with no training.
Optionally the same firmware **speaks** replies with an on-device neural voice.

You'll want it if you need voice input that keeps working offline, keeps audio private, or
can't depend on a server. For example:

- **Radios without keyboards.** A LoRa node (Meshtastic, or your own protocol) with a
  microphone but no keyboard: speak a message, and the node sends it as text. A spoken
  sentence becomes a few dozen bytes of text; even a very low-bitrate voice codec (Codec2
  at 700 bit/s) needs about ten times that, which matters on a link that moves a few
  hundred bytes per second at best.
- **Speakers without screens.** The receiving node reads incoming text messages aloud.
- **Voice control that works when the WiFi doesn't**, and never sends audio anywhere.

Skip it if you need very high accuracy, languages other than English, or it has to run on
a smaller chip (see below).

(This may also be the only project that involves both **LoRa** radios and **LoRA**,
Low-Rank Adaptation. The original research plan in `docs/historical/` explored LoRA-style
expert adapters for the speech model. They aren't used yet.)

## What you need

- An **ESP32-S3 with 16 MB flash and 8 MB PSRAM** (e.g. an "N16R8" module). The speech
  model alone is 6 MB, and the runtime needs PSRAM for its working memory.
- A microphone (and a speaker for text to speech). Everything was built and tested on the
  **Waveshare ESP32-S3-CAM** (ES7210 mic, ES8311 speaker codec), and the examples use its
  pins and drivers. On another board, give `stt` 16 kHz PCM from whatever mic you have.
- To build: Linux or WSL, [ESP-IDF v5.5.1](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/get-started/),
  and [uv](https://docs.astral.sh/uv/) for the Python tools.

## How it works

```
mic -> 16 kHz audio -> log-mel features -> Citrinet-256 (neural net) -> letters/word pieces -> text
                                                                      \-> wake phrase score
text -> pronunciation dictionary -> phonemes -> sanoTTS voice -> 24 kHz audio -> speaker
```

- **Speech to text** uses NVIDIA's [Citrinet-256](https://huggingface.co/nvidia/stt_en_citrinet_256_ls),
  a 9.8-million-parameter recognizer trained on 960 hours of English audiobooks. It
  outputs text directly ("CTC"), so it has no vocabulary list to maintain.
- **Wake phrases**: the same model output is scored against your phrase's spellings, so
  any phrase works. `export/kws_validate.py` checks a phrase against hours of other speech
  to pick a threshold that avoids false wakes.
- **Text to speech** uses the 294k-parameter "nano" voice from
  [sanoTTS](https://github.com/Ampixa/sanoTTS). It is optional: the firmware builds
  without it.

### MMRT, the runtime

The model runs on **MMRT** (`mmrt/`, the "micromodels runtime"), a small inference engine
written for BabyTalk and meant to host other small models later.
We started with Espressif's ESP-DL. It gave correct results but took 1.33 s to process
each second of audio, because it re-read every layer's weights from flash for every frame
of audio and used one core. Fixing that meant patching ESP-DL, which nobody else could easily
build, so we wrote our own. MMRT is about 13 times faster than stock ESP-DL and half its
size, with identical output:

- It reads the model **in place from flash**; nothing is copied to RAM at startup.
- It copies each layer's weights into fast internal RAM **once per layer**, not once per
  frame, and splits the work across **both cores**.
- Its kernels are **hand-written assembly** for the ESP32-S3's vector instructions,
  doing 16 multiply-adds per instruction.
- It **fuses layers** so intermediate results stay in internal RAM, and on short clips it
  overlaps copying the next layer's weights with computing the current one.
- It loads **8-bit or 4-bit weights**, and every kernel is tested bit for bit against a
  plain C version.

Details, a C usage example and the measurements behind each design choice:
[`mmrt/README.md`](mmrt/README.md).

### Quantization: fitting a 40 MB model into 6 MB

The original model uses 32-bit floats (~39 MB). It is compressed twice:

1. **int8**: weights and activations become 8-bit integers (9.8 MB).
2. **int4**: each output channel keeps 16 weight values in a small lookup table, chosen
   with GPTQ (an error-compensating method). The first and last layers stay 8-bit.
   Result: **6.0 MB**.

Both are ready to use in [`models/`](models/). Word error rate on LibriSpeech test-clean
(clean read English), lower is better:

| model | file | size | word error rate |
|---|---|---|---|
| original (float) | | ~39 MB | 3.8% |
| int8 | `models/citrinet256_int8.mmrt` | 9.8 MB | 6.3% |
| **int4 (default)** | `models/citrinet256_int4.mmrt` | **6.0 MB** | **8.2%** |

Real microphones in real rooms do worse than clean recordings; expect some misheard words.

Your MicroPython code is the same for either model; the file records its own format.
The difference is flash space: the int4 model fits the standard 6 MB model partition, and
the int8 model needs the "int8 layout" (10 MB partition, see Install).

```python
import stt
import model_tools                          # mpy/examples/model_tools.py

model_tools.info()
# int4: {'format': 'int4', 'bytes': 5991232, 'int4_layers': 146, 'partition_bytes': 6291456}
# int8: {'format': 'int8', 'bytes': 9776192, 'int4_layers': 0, 'partition_bytes': 10485760}

text = stt.transcribe(pcm)                  # same call either way
```

### Loading models from an SD card

The runtime reads the model directly from **flash**, where it can be memory-mapped and
read at ~32 MB/s. An SD card can't be memory-mapped, and every transcription reads the
whole model, so running straight from SD would be much slower. (We haven't measured SD
speed on this board.)

What does work is keeping models on an SD card, or any MicroPython filesystem, and
installing one into flash from MicroPython:

```python
import machine, os, model_tools
os.mount(machine.SDCard(), "/sd")           # pins depend on your board
model_tools.install("/sd/citrinet256_int4.mmrt")   # writes, verifies, resets the board
```

Tested from the board's internal filesystem (not an SD card): installing and verifying the
6 MB int4 model took 48 seconds.

## Performance and memory

Measured on the Waveshare ESP32-S3-CAM, from MicroPython, int4 model. Times include the
audio feature step.

**Speed**

| task | time |
|---|---|
| Transcribe 1 s of speech | 0.35 s |
| Transcribe 2 s of speech | 0.51 s |
| Transcribe 4 s of speech | 0.83 s |
| Transcribe 10 s of speech | 1.72 s |
| Transcribe 1 s, with the 6 MB weight cache | 0.31 s |
| Transcribe 4 s, with the 6 MB weight cache | 0.79 s |
| Speak 1 s of speech | 0.26 s |

The optional weight cache (`stt.cache(6144)`) leaves little PSRAM free, so with it, clips
are limited to about 4 s under MicroPython.

**Flash (16 MB)**

| item | size |
|---|---|
| Firmware (total) | 3.6 MB |
| of which MicroPython and the camera module | 1.7 MB |
| of which text to speech (voice 0.34 MB, pronunciation dictionary 1.5 MB) | 1.9 MB |
| Speech model, int4 | 6.0 MB |
| Speech model, int8 (int8 layout) | 9.8 MB |
| Left for your files, int4 layout | 6 MB |
| Left for your files, int8 layout | 2 MB |

**PSRAM (8 MB)**

| item | size |
|---|---|
| Free when idle, speech loaded | 7.9 MB |
| Used while transcribing 4 s | 0.2 MB |
| Used by a 4 s spoken reply | 0.4 MB |
| Optional weight cache | 6 MB |

**Internal RAM** (the scarce one)

| item | size |
|---|---|
| Shared block for listening / speaking (they take turns) | 84.5 KB |
| Text to speech, permanent buffers | 104 KB |
| Free for your program | ~42 KB |

Bluetooth is disabled in this firmware to make that room.

## Install

**1. Build the firmware** (MicroPython v1.27.0 with a small patch that lets `machine.I2S`
drive the codec's master clock; more options in [`mpy/README.md`](mpy/README.md)):

```bash
REPO=$PWD                                   # this repository
. ~/esp/esp-idf/export.sh                   # wherever your ESP-IDF v5.5.1 lives
mkdir -p ~/build/sentry-fw && cd ~/build/sentry-fw
git clone --depth 1 -b v1.27.0 https://github.com/micropython/micropython
git -C micropython apply $REPO/mpy/patches/micropython-i2s-mck.patch
make -C micropython/ports/esp32 submodules
git clone https://github.com/Ampixa/sanoTTS ~/build/tts/sanoTTS             # optional: text to speech,
git -C ~/build/tts/sanoTTS checkout 18e26b2b365bff41d211e516b0760021451438f1  # at the audited commit
cd $REPO && mpy/build.sh
```

**2. Flash** the firmware and a model, then clear the filesystem area once:

```bash
# int4 model (default layout)
esptool.py --chip esp32s3 write_flash 0x0 ~/build/sentry-fw/out/firmware-stt.bin \
  0x410000 models/citrinet256_int4.mmrt
esptool.py --chip esp32s3 erase_region 0xA10000 0x5F0000

# or: int8 model (int8 layout: same firmware, bigger model partition)
esptool.py --chip esp32s3 write_flash 0x0 ~/build/sentry-fw/out/firmware-stt.bin \
  0x410000 models/citrinet256_int8.mmrt
esptool.py --chip esp32s3 write_flash 0x8000 ~/build/sentry-fw/out/partition-table-int8.bin
esptool.py --chip esp32s3 erase_region 0xE10000 0x1F0000
```

**3. Try it**: press Enter, talk, and the board says back what it heard.

```bash
uv run --with mpremote tools/echo.py
```

`mpy/examples/` also has a wake-phrase demo, a speaker demo and `model_tools.py`. The full
MicroPython API is in [`mpy/README.md`](mpy/README.md).

**Building the models yourself** (optional; this is how `models/` was made). Download the
NVIDIA checkpoint and LibriSpeech, then quantize (the int4 step takes ~15 minutes):

```bash
mkdir -p data/models/citrinet_256_ls && curl -L -o data/models/citrinet_256_ls/stt_en_citrinet_256_ls.nemo \
  https://huggingface.co/nvidia/stt_en_citrinet_256_ls/resolve/main/stt_en_citrinet_256_ls.nemo
# LibriSpeech dev-clean and test-clean (https://www.openslr.org/12) unpacked into data/librispeech/
cd export
uv run export_onnx.py --cle && uv run mmrt_quant.py && uv run mmrt_export.py    # int8
uv run int4_gptq.py --keep-io                                                    # int4
uv run mmrt_cb4.py ../data/models/mmrt/citrinet256_cb4_gptq16_io.mmrt -o ../data/models/mmrt/citrinet256_int4.mmrt
```

## Training for the real world (work in progress)

The speech model learned from clean audiobooks. At a desk it does well, but with an engine,
music or people talking in the background, it falls apart: on our recordings over a diesel
truck idling, even the full-precision model got about half the words wrong. So we record
real-world audio on the board itself and fine-tune the model on it.

**Recording** (`field/`): a session plays a background sound (a TV, a coffee-shop loop, a
truck engine, or the real thing) and shows sentences on screen. You read each one aloud,
then the laptop reads the same sentence in one of ~35 synthetic voices at a random speed,
and the board records both over WiFi. Sentences come from audiobooks, today's news,
popular Wikipedia articles, Hacker News headlines, and your own list of words
(`field/terms.txt`: names, commands, jargon). Background-only recordings are saved too,
for mixing into training.

**Measuring** (`export/field_eval.py`): one command scores every model on the recordings.
A fifth of the sentences are reserved for testing and never used in training.
`export/hard_words.py` keeps a running list of the words the model gets wrong, and the
recorder can target them (`--hard`) or redo the sentences it missed (`--retake`).

**Training** (`train/finetune.py`, ~15 minutes on a laptop GPU): the model keeps learning
from 100 hours of audiobooks, plus synthetic speech of modern text and the field
recordings, with the recorded background noise and other damage (muffling, clipping,
dropouts) mixed in. Blending the fine-tuned weights with the original ones lets us choose
how much clean-speech accuracy to keep.

First results, on 120 recordings none of the models had heard (a new noise, and 30 clips
of a real person reading), word error rate:

| model (full precision) | new recordings | clean audiobooks |
|---|---|---|
| original | 46% | 3.9% |
| fine-tuned, blended 70% with the original | 25% | 4.9% |
| fine-tuned | 24% | 6.2% |

So far this is the full-precision model on the PC. The int8 and int4 versions for the board
still need to be rebuilt from it.

```bash
cd field && uv run prompts.py                                   # build the sentence pool
uv run capture.py --condition truck-idle --noise "diesel idling"   # a recording session
cd ../export && uv run field_eval.py --split all                # score the models
cd ../train && uv run make_tts.py && uv run finetune.py         # fine-tune on the GPU
```

## BabyTalk for AtomVM (Erlang / Elixir)

The same engine, from Erlang and Elixir on AtomVM v0.7 ([atomvm/README.md](atomvm/README.md)):
speech to text, text to speech, a wake phrase with audible chimes, microphone and speaker,
run by a supervised `gen_server`, plus **MMRT** as a standalone library of int8/int4 vector kernels
(matvec, matmul, dot, top-k... on the S3's SIMD unit) for any AtomVM project. Transcripts are
identical to the MicroPython firmware's. AtomVM has no I2S driver yet, so BabyTalk brings its
own C drivers for the board's audio chips (ES7210 mic ADC, ES8311 codec, NS4150B amp,
CH32V003 IO expander).

```erlang
{ok, Pcm} = babytalk:record(3),
{ok, Text, _Info} = babytalk:transcribe_sync(Pcm, 10000),
ok = babytalk:speak([<<"You said: ">>, Text]).
```

## Limitations

- English only, 16 kHz audio. Transcription starts after you stop talking (no live
  streaming), and it is best with one speaker at a time close to the mic.
- The models on the board are still the original ones, which struggle with loud background
  noise; the fine-tuned model (above) isn't quantized for the board yet.
- The int4 model trades ~2 points of accuracy for size (see the table above).
- The tiny voice is clear but clearly synthetic, and mispronounces some words.
- Tested on one board. Another ESP32-S3 board needs its own pins and audio codec driver.
- Listening and speaking take turns (they share one block of internal RAM).

## Future work

- Rebuild the board's int8 and int4 models from the fine-tuned model, then train with the
  8- and 4-bit arithmetic simulated (quantization-aware training) to recover their losses.
- Record in a real vehicle, and on the target device (a LilyGO T-Watch S3 Plus).
- Transcribe while you are still talking (streaming).
- A better voice: larger sanoTTS voices, or a voice distilled from bigger TTS models.
- Free more internal RAM so the camera, Bluetooth and speech can all run together.
- Prebuilt firmware downloads; more boards; more languages.

## See also

- [ESP-SR](https://github.com/espressif/esp-sr): Espressif's wake words (WakeNet) and
  fixed-command recognition (MultiNet) for ESP32-S3.
- [ESP-DL](https://github.com/espressif/esp-dl): Espressif's general neural network
  library for ESP32 chips.
- [TensorFlow Lite Micro "micro_speech"](https://github.com/tensorflow/tflite-micro) and
  [Edge Impulse](https://edgeimpulse.com/): train your own keyword spotter.
- [sanoTTS](https://github.com/Ampixa/sanoTTS): the tiny neural TTS used here
  (Arduino, ESPHome, browser).
- [TinyTTS](https://github.com/pschatzmann/TinyTTS) and
  [rvTTS](https://github.com/ArmstrongSubero/rvTTS): other neural TTS on microcontrollers.
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp): Whisper speech recognition for
  PCs, phones and single-board computers (too large for an ESP32).
- [Meshtastic](https://meshtastic.org/): open LoRa mesh messaging.
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo): where Citrinet comes from.

## Credits and licenses

BabyTalk was made by **Thomas Lackner** and **Claude Opus 5.5** (Anthropic's AI model),
working together.

The code in this repository is released under the [MIT license](LICENSE), copyright
Thomas Lackner. Other people's work used here keeps its own license:

- **Speech models** (`models/`): derived from NVIDIA's
  [stt_en_citrinet_256_ls](https://huggingface.co/nvidia/stt_en_citrinet_256_ls),
  CC-BY-4.0. Converted and quantized by this project; see [`models/README.md`](models/README.md).
- **Text to speech**: [sanoTTS](https://github.com/Ampixa/sanoTTS) by Ampixa, fetched at build
  time from your own checkout (pinned commit) -- none of it is in this repository. Its runtime
  is MIT and its dictionary (from misaki) Apache-2.0, but three of the files we compile carry
  no licence of their own and fall under sanoTTS's GPL-3.0 default, so **firmware binaries
  that include text to speech should be treated as GPL-3.0** until that is clarified
  upstream. File-by-file details: [`components/sanotts/LICENSES.md`](components/sanotts/LICENSES.md).
- **MicroPython** (MIT) and **ESP-IDF** (Apache-2.0), downloaded at build time; the
  firmware also compiles in Espressif's **esp-dsp** (Apache-2.0, the log-mel front end's
  FFT). The `bench/` firmware uses **esp-nn** (Apache-2.0).
- **AtomVM** (Apache-2.0 OR LGPL-2.1-or-later): the VM and its Erlang/Elixir standard
  libraries (`boot.avm`) are part of the AtomVM firmware image (`atomvm/`). Downloaded at
  build time; our local patch is in `atomvm/patches/`.
- **Audio codec drivers**: the ES7210 and ES8311 register sequences in `mpy/drivers/`,
  `stt/main/mic.c`, `bench/main/bench_rec.c` and `atomvm/components/atomvm_babytalk/
  board_audio.c` are derived from Espressif's esp-bsp drivers (Apache-2.0,
  [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)); each file says so.
- Parts of the 1x1 convolution kernel are adapted from Espressif's ESP-DL (MIT).

Project history and early design notes: `docs/historical/`.

---

## Appendix: ESP32-S3 benchmarks (Waveshare ESP32-S3-CAM)

Measured with the `bench/` app on an ESP32-S3 (rev v0.2, 240 MHz) with 16 MB QIO flash at
80 MHz, 8 MB octal PSRAM at 80 MHz, and a 64 KB data cache. These are properties of the
chip and memory, so they may help with other projects. Terms are explained in the
glossary below.

**Memory speed**

| memory | read | write |
|---|---|---|
| Internal SRAM | 425 MB/s (simple C loop; vector loads go faster) | 760 MB/s |
| PSRAM, streaming | 88 MB/s | 49 MB/s |
| Flash, memory-mapped | 32 MB/s | |
| Flash, `esp_partition_read()` | 12.7 MB/s | |
| SD card | not measured | |

Flash and PSRAM share one bus, so their traffic takes turns: reading from both at once is
no faster than reading from one. A second CPU core does not add memory bandwidth.

**Free memory with bare ESP-IDF** (no WiFi): internal RAM 286 KB of 316 KB free (largest
block 270 KB), PSRAM 8.0 MB free.

**Arithmetic** (8-bit multiply-adds, in GMAC/s)

| setup | GMAC/s |
|---|---|
| Theoretical vector peak, one core (16 per cycle at 240 MHz) | 3.84 |
| MMRT 1x1 kernel, one core, weights in SRAM or cached | ~3.2 |
| Generic dot-product kernel called per row (ESP-NN), one core, SRAM | 0.68 |
| Same, two cores, SRAM | 1.35 |
| Weights streamed from PSRAM, each used once | 0.085 |
| Weights in PSRAM, each reused for 64 frames, two cores | 1.18 |
| Weights read in place from flash, reused for 64 frames, two cores | 0.95 |

The lesson: on this chip, speed comes from **reusing each weight many times once it is
in fast memory**. Streaming weights once per use caps you at the memory speed (88 MB/s of
8-bit weights is 0.088 GMAC/s), no matter how fast the math units are.

**MMRT kernels:** the 1x1 convolution runs at 1.19 CPU cycles per 16-wide vector
multiply-add, with operands in SRAM or cached PSRAM. The 4-bit weight unpacking runs at
5.2 cycles per weight (portable C: 9.3).

**Glossary**

- **MAC / GMAC/s**: a multiply-accumulate is one multiplication added to a running sum, the
  basic step of a neural network. GMAC/s is billions of them per second.
- **SRAM** (internal RAM): the ~512 KB of fast memory inside the chip, shared with
  ESP-IDF, WiFi and MicroPython.
- **PSRAM**: the external 8 MB RAM chip; bigger and slower than SRAM.
- **Memory-mapped flash**: flash the CPU reads like RAM, through a cache, without copying.
- **PIE / SIMD**: the ESP32-S3's vector instructions, which do 16 8-bit operations at once.
- **int8 / int4 / quantization**: storing a network's numbers as 8- or 4-bit integers
  instead of 32-bit floats: smaller and faster, slightly less accurate.
- **GPTQ**: a quantization method that corrects each rounding error using the remaining
  weights, which keeps accuracy at 4 bits.
- **CTC**: the output style of the speech model: one letter or word piece (or "nothing")
  per 80 ms of audio, merged into text.
- **Word error rate**: the fraction of words a transcript gets wrong (substituted, missing
  or extra).
- **LoRa / LoRA**: LoRa is a long-range, very low-bandwidth radio. LoRA (Low-Rank
  Adaptation) is a way to adapt a neural network by adding small trainable matrices.
