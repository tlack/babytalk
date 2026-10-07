# BabyTalk from C (ESP-IDF components)

The speech engines as ESP-IDF components, for your own ESP-IDF app. The MicroPython
(`mpy/`) and AtomVM (`atomvm/`) firmwares are thin bindings over these same components, so
all three give the same transcripts.

```c
#include "sram_pool.h"
#include "stt_engine.h"
#include "tts_engine.h"

sram_pool_reserve();                          // early in app_main, before internal RAM fragments
stt_engine_open();                            // maps the "model" partition (0 on success)

stt_result_t r;
stt_engine_run(pcm, n_samples, &r);           // 16 kHz mono int16 -> r.text

int16_t *out; int n_out;
if (tts_say(r.text, 0.8f, &out, &n_out, NULL) == 0) {
    play(out, n_out);                         // 24 kHz mono int16, your I2S driver
    heap_caps_free(out);
}
```

| component | what | needs |
|---|---|---|
| `stt_engine` | speech to text and wake phrases: log-mel front end, Citrinet-256, CTC decoding | `mmrt`, `sram_pool`, esp-dsp; a `model` partition |
| `mmrt` | the int8/int4 inference runtime ([`../mmrt/README.md`](../mmrt/README.md)) | ESP32-S3 (PIE assembly), ESP32-P4 (vector unit), or portable C on any chip |
| `sanotts` | text to speech (sanoTTS), optional | `sram_pool`; a sanoTTS checkout at build time |
| `sram_pool` | one 84.5 KB internal-RAM block that the engines take turns using | |

## Adding them to a project

The components refer to sources elsewhere in this repository (`../mmrt/`, `../stt/main/`),
so point at this directory rather than copying it:

```cmake
# your project's CMakeLists.txt, before project()
set(EXTRA_COMPONENT_DIRS "/path/to/babytalk/components")
```

and `REQUIRES stt_engine sram_pool sanotts` (leave out `sanotts` for speech to text only)
in your main component.

**Text to speech** is built only when you pass sanoTTS's sources. Copy and patch them from
a sanoTTS checkout at the audited commit (the script refuses any other; see
[`sanotts/LICENSES.md`](sanotts/LICENSES.md)):

```bash
git clone https://github.com/Ampixa/sanoTTS ~/build/tts/sanoTTS
git -C ~/build/tts/sanoTTS checkout 18e26b2b365bff41d211e516b0760021451438f1
components/sanotts/prepare.sh ~/build/tts/sanoTTS build/sanotts_src     # [nano|heart|heart4]
idf.py -D SANOTTS_SRC=$PWD/build/sanotts_src build
```

Without `SANOTTS_SRC`, `sanotts` builds as a stub whose `tts_say()` returns -4, so a
speech-to-text-only app links unchanged. A firmware that includes text to speech should be
treated as GPL-3.0 if you distribute it ([`sanotts/LICENSES.md`](sanotts/LICENSES.md)).

Build options:

| option | default | what |
|---|---|---|
| `-D SANOTTS_SRC=dir` | unset (stub) | sanoTTS sources from `prepare.sh` |
| `-D SANOTTS_BSS_PSRAM=1` | off | sanoTTS's ~104 KB of static buffers in PSRAM (needs `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY`) |
| `CONFIG_SRAM_POOL_IN_PSRAM` | n | the shared 84.5 KB block in PSRAM: for boards short of internal RAM; a little slower on octal PSRAM, more on quad |

**sdkconfig:** PSRAM enabled (`CONFIG_SPIRAM=y`); the firmwares here use octal PSRAM at
80 MHz on the S3 and 200 MHz on the P4 (see `atomvm/sdkconfig.babytalk*`).

**Partitions:** a data partition labelled `model` (subtype `0x40`), at least 6 MB for the
int4 model or 10 MB for int8. Flash `../models/citrinet256_int4.mmrt` to its offset. The
larger P4 voices (`heart`, `heart4`) also need a `voice` partition for `voice.bin`
([`../docs/TTS_VOICES.md`](../docs/TTS_VOICES.md)).

```
# Name,   Type, SubType, Offset,   Size
nvs,      data, nvs,     0x9000,   0x6000
phy_init, data, phy,     0xf000,   0x1000
factory,  app,  factory, 0x10000,  4M
model,    data, 0x40,    0x410000, 6M
```

## Speech to text: `stt_engine.h`

The model is read in place from flash; nothing is copied to RAM at open. Inference uses both
cores: the calling task (or the engine's background task on core 1) and MMRT's worker on
core 0. Return codes: 0 success, -1 no `model` partition, not open yet, or busy, -2 not a
model image (or no background result), -3 out of memory, -4 a clip too short or too long.

| function | |
|---|---|
| `int stt_engine_open(void)` | map the `model` partition and start the workers; call before the first run |
| `void stt_engine_close(void)` | release the cache, buffers and phrase, e.g. before another engine needs internal RAM; open again before the next run |
| `void stt_engine_info(stt_info_t *)` | model size and format, cache size, internal and PSRAM free |
| `int stt_engine_run(const int16_t *pcm, int n, stt_result_t *r)` | transcribe `n` samples of 16 kHz mono PCM, blocking |
| `int stt_engine_run_ch(pcm, n, channels, r)` | the same for interleaved PCM (`n` frames); uses channel 0, e.g. the left slot of a stereo I2S capture |
| `int stt_engine_start(pcm, n, channels)` | copy the PCM and transcribe it on the engine's task |
| `int stt_engine_busy(void)` | whether a background run is still going |
| `int stt_engine_take(stt_result_t *r)` | the background run's result and return code (-2: none) |
| `int stt_engine_set_phrase(const char *const *spellings, int n)` | a wake phrase, as one or more spellings; every later run scores it. `n = 0` clears |
| `size_t stt_engine_cache(size_t bytes)` | keep up to `bytes` of weights in PSRAM (faster than flash; 0 releases) |
| `void stt_engine_act_sram(size_t max, size_t reserve)` | keep activations up to `max` bytes in internal RAM while `reserve` bytes stay free (0: all in PSRAM) |

`stt_result_t` holds `text` (up to 1024 bytes), `score` (the wake-phrase score: 0 means the
phrase is the best reading of the audio, more negative is a worse match; -1e30 with no phrase),
`kw_start` / `kw_end` (the matched span, in 80 ms output frames), `frames`, and the time spent
in the front end, the model and the decoder (`fe_ms`, `model_ms`, `dec_ms`).

Wake-phrase spellings: only letters count ("tomato face" equals "tomatoface"); give several
when a phrase can be heard more than one way. To choose a threshold that avoids false wakes,
run `export/kws_validate.py` on the PC.

## Text to speech: `tts_engine.h`

sanoTTS's 294k-parameter "nano" voice by default: a pronunciation dictionary (no espeak),
then the voice, at 24 kHz.

| function | |
|---|---|
| `int tts_say(const char *text, float volume, int16_t **pcm, int *n, tts_stats_t *st)` | synthesize `text`; `*pcm` is allocated in PSRAM (free with `heap_caps_free`), peak-normalized to `volume` (0..1). `st` may be NULL |
| `void tts_set_length_scale(float)` | speaking pace for later calls: 1.0 the voice's own, 1.1 is 10% slower (0.5 to 2.0) |
| `int tts_reserve(void)` | reserve the shared block now (-2: no block) |
| `void tts_release(void)` | give up the shared block (`tts_say` already does after each utterance) |

Return codes: 0 success, -1 the text couldn't be turned into phonemes, -2 no block for the
synthesizer's working memory, -3 out of PSRAM, -4 built without sanoTTS, others from the
synthesizer. `tts_stats_t` reports phonemes, frames, samples and the time spent
(`synth_ms / (samples / 24000)` is the real-time factor).

## Sharing memory: `sram_pool.h`

Speech to text and text to speech each need one large contiguous block of internal RAM, and
once something small lands in the middle of free memory, no large block comes back. So they
share a single 84.5 KB block, reserved once, and take turns: when one engine asks for it, the
other is told to let go. Your app only needs to reserve it early:

| function | |
|---|---|
| `int sram_pool_reserve(void)` | take the block now (0 on success); call early in `app_main` |
| `void *sram_pool_acquire(int owner, sram_pool_evict_fn evict)` | the block for `owner`, evicting the current holder (whose `evict` returns 0 to give it up, nonzero to refuse) |
| `void sram_pool_release(int owner)` | give it up |

You can also use the block for your own work between speech calls, through the same
acquire / release calls with your own owner id.

## Audio

These components take and return PCM; they don't drive a microphone or speaker. Starting
points for your own driver: `../stt/main/mic.c` (ES7210 capture on the Waveshare
ESP32-S3-CAM), and `../atomvm/components/atomvm_babytalk/board_audio.c` (ES7210 and ES8311,
pins chosen at run time, behind the small interface in `board_audio.h`).
