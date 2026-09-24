# esp32-micromodels: offline speech to text (and back) on an ESP32-S3

An ESP32-S3 that **understands what you say and talks back, with no cloud and no
internet**, driven from MicroPython:

```python
import stt, tts
text = stt.transcribe(pcm)                 # 16 kHz audio -> "what's the weather like today"
audio = tts.say("You said: " + text)       # text -> 24 kHz audio for the speaker
```

## Is this for you?

Most "voice on ESP32" projects do one of two things:

- **send the audio to a cloud service** (Whisper API, Google, wit.ai) and wait for text, or
- **recognize a fixed list of commands or wake words** that were trained in advance
  (e.g. ESP-SR's MultiNet, Edge Impulse / TensorFlow Lite Micro keyword spotting).

This project is different: a **general-purpose English speech recognizer** that runs
entirely on the chip. It writes down *any* sentence, with no training and no command list.
A wake phrase is any phrase you type ("wake up, tomato face"), also with no training.
Optionally the same firmware **speaks** replies with an on-device neural voice.

You'll want it if you need voice input that keeps working offline, keeps audio private, or
can't depend on a server. Skip it if you need very high accuracy, languages other than
English, or it has to run on a smaller chip (see requirements).

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

- **Speech to text** uses NVIDIA's [Citrinet-256](https://huggingface.co/nvidia/stt_en_citrinet_256_ls)
  (CC-BY-4.0), a 9.8-million-parameter recognizer trained on 960 hours of English
  audiobooks. It outputs text directly ("CTC"), so it has no vocabulary list to maintain.
- **Wake phrases**: the same model output is scored against your phrase's spellings, so
  any phrase works. `export/kws_validate.py` checks a phrase against hours of other speech
  to pick a threshold that avoids false wakes.
- **Text to speech** uses the 294k-parameter "nano" voice from
  [sanoTTS](https://github.com/Ampixa/sanoTTS) (runtime MIT, dictionary Apache-2.0). It is
  optional: the firmware builds without it.

### MMRT, the runtime

Off-the-shelf options were too slow or too big, so the model runs on **MMRT** (`mmrt/`), a
small inference engine written for this project:

- It reads the model **in place from flash**; nothing is copied to RAM at startup.
- It uses the ESP32-S3's vector instructions (hand-written assembly) and **both cores**.
- It loads **int8 or 4-bit weights** from a simple model file. Its output matches a PC
  simulation of the same model bit for bit, which is how every kernel is tested.

### Quantization: fitting a 40 MB model into 6 MB

The original model uses 32-bit floats (~39 MB). Here it is compressed twice:

1. **int8**: weights and activations become 8-bit integers (9.8 MB).
2. **4-bit weights**: each output channel keeps 16 weight values in a small lookup table,
   chosen with GPTQ (an error-compensating method). The first and last layers stay int8.
   Result: **6.0 MB**.

Each step costs some accuracy. Word error rate on LibriSpeech test-clean (clean read
English), lower is better:

| model | size | word error rate |
|---|---|---|
| original (float) | ~39 MB | 3.8% |
| int8 | 9.8 MB | 6.3% |
| **4-bit, used here** | **6.0 MB** | **8.2%** |

Real microphones in real rooms do worse than clean recordings; expect some misheard words.

## Performance and memory (Waveshare ESP32-S3-CAM, MicroPython)

| | |
|---|---|
| Transcribe 1 s / 4 s / 10 s of speech | ~0.27 / 0.56 / 1.07 s (plus ~0.05 s per second for audio features) |
| Speak | ~0.26 s of compute per second of speech |
| Flash | firmware 3.6 MB (MicroPython 1.7 MB, TTS 1.9 MB), model 6 MB, ~6 MB left for files |
| PSRAM | ~8.3 MB free when idle; a 4 s transcription uses ~0.2 MB |
| Internal RAM | the tight one: ~42 KB left free. Bluetooth is disabled to make room |

An optional 6 MB PSRAM cache (`stt.cache(6144)`) makes transcription 5-10% faster.

## Install

Prebuilt images aren't published yet, so for now you build them (details: `mpy/README.md`).

**1. Build the speech model** (downloads the NVIDIA model and LibriSpeech, then quantizes;
the 4-bit step takes ~15 minutes on a desktop):

```bash
mkdir -p data/models/citrinet_256_ls && curl -L -o data/models/citrinet_256_ls/stt_en_citrinet_256_ls.nemo \
  https://huggingface.co/nvidia/stt_en_citrinet_256_ls/resolve/main/stt_en_citrinet_256_ls.nemo
# LibriSpeech dev-clean and test-clean (https://www.openslr.org/12) unpacked into data/librispeech/
cd export
uv run export_onnx.py --cle && uv run mmrt_quant.py && uv run mmrt_export.py
uv run int4_gptq.py --keep-io
uv run mmrt_cb4.py ../data/models/mmrt/citrinet256_cb4_gptq16_io.mmrt -o ../data/models/mmrt/citrinet256_int4.mmrt
cd ..
```

**2. Build the firmware** (MicroPython v1.27.0 with a small patch that lets `machine.I2S`
drive the codec's master clock):

```bash
REPO=$PWD                                   # this repository
. ~/esp/esp-idf/export.sh                   # wherever your ESP-IDF v5.5.1 lives
mkdir -p ~/build/sentry-fw && cd ~/build/sentry-fw
git clone --depth 1 -b v1.27.0 https://github.com/micropython/micropython
git -C micropython apply $REPO/mpy/patches/micropython-i2s-mck.patch
make -C micropython/ports/esp32 submodules
git clone --depth 1 https://github.com/Ampixa/sanoTTS ~/build/tts/sanoTTS    # optional: text to speech
cd $REPO && mpy/build.sh
```

**3. Flash** (firmware, model, then clear the filesystem area once):

```bash
esptool.py --chip esp32s3 write_flash 0x0 ~/build/sentry-fw/out/firmware-stt.bin \
  0x410000 data/models/mmrt/citrinet256_int4.mmrt
esptool.py --chip esp32s3 erase_region 0xA10000 0x5F0000
```

**4. Try it**: press Enter, talk, and the board says back what it heard.

```bash
uv run --with mpremote tools/echo.py
```

`mpy/examples/` also has a wake-phrase demo and a speaker demo. The full MicroPython API
is in `mpy/README.md`.

## Limitations

- English only, 16 kHz audio. Transcription starts after you stop talking (no live
  streaming), and it is best with one speaker at a time close to the mic.
- The 4-bit model trades ~2 points of accuracy for size (see the table above).
- The tiny voice is clear but clearly synthetic, and mispronounces some words.
- Tested on one board. Another ESP32-S3 board needs its own pins and audio codec driver.
- Listening and speaking take turns (they share one block of internal RAM).

## Future work

- Recover the 4-bit accuracy loss by fine-tuning with 4-bit weights in place.
- Transcribe while you are still talking (streaming).
- A better voice: larger sanoTTS voices, or a voice distilled from bigger TTS models.
- Free more internal RAM so the camera, Bluetooth and speech can all run together.
- Prebuilt firmware and model downloads; more boards; more languages.

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
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo): where Citrinet comes from.

Project history and early design notes: `docs/historical/`.
