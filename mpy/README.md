# MicroPython `stt` — on-device speech to text for MicroPython

The runtime (mmrt), front end, wake-phrase scorer and a small engine, packaged as an IDF
component (`components/stt_engine`) plus a thin MicroPython user C module (`stt/`).
The model is read in place from a `model` flash partition. Built and verified on the
Waveshare ESP32-S3-CAM (ESP32-S3R8: 8MB octal PSRAM, 16MB flash) alongside the
optional camera API + mp_jpeg modules (github.com/cnadler86/micropython-camera-API).

```python
import stt
stt.transcribe(pcm)              # 16 kHz int16 bytes -> "what's the weather like today"
stt.transcribe(stereo, 2)        # interleaved I2S capture: uses the left slot
stt.phrase(["wake up tomato face", ...])   # score every run for a wake phrase
stt.last()                       # {'text', 'score', 'span', 'frames', 'fe_ms', 'model_ms', 'dec_ms'}

# background (keeps asyncio running): poll every >= 20 ms
stt.start(pcm, 2)
while stt.busy(): await asyncio.sleep_ms(20)
text = stt.result()

stt.cache(6144)                  # optional: up to 6MB of weights in PSRAM (0 releases)
stt.act_sram(32, 40)             # optional: activations <= 32KB in SRAM, keep 40KB free
stt.info()                       # model size / format, cache, internal + PSRAM free
stt.rms(pcm, 2)                  # level of a buffer (silence trimming, VAD)
stt.close()                      # give back stt's memory (reopens on demand)
```

## Text to speech: `tts` (sanoTTS)

```python
import tts
tts.reserve()                    # early (boot.py): reserve the shared 84KB internal SRAM pool
pcm = tts.say("Hello! I am a tiny computer.", 0.77, True)   # int16 @ 24 kHz, peak 0.77, stereo
tts.last()                       # phonemes, frames, samples, g2p_ms, synth_ms, rtf, arena_peak
```

[sanoTTS](https://github.com/Ampixa/sanoTTS)'s `en_us_e12nano` voice (294k params, int8,
24 kHz) with its espeak-free lexicon G2P, built from the repo's ESPHome component
(`build.sh` copies it from `SANOTTS_DIR`, default `~/build/tts/sanoTTS`, which must be at
the audited commit, and patches it to one scratch bank). Licensing: runtime MIT, dictionary
Apache-2.0, but three compiled files have no licence of their own -- treat firmware binaries
with `tts` as GPL-3.0; see `components/sanotts/LICENSES.md`. The
firmware builds without `tts` if that checkout is missing. Measured here: **RTF 0.25-0.26**
(4.06 s of speech in 1.05 s), all MACs on the PIE kernels, arena peak 84,208 B = the
project's reference. `examples/say_demo.py` plays it through the ES8311 + PA.

Flash: the gold dictionary is 1.5 MB (the full one with "silver" words 3.05 MB), the voice
340 KB. **Internal SRAM is the constraint**: the synthesizer needs one contiguous 84 KB
block, and so does mmrt's staging (72 KB). Both come from `components/sram_pool`: one block
reserved at startup and handed to whichever engine runs (stt is evicted when idle and
re-acquires on its next run). Listening and speaking alternate on this board anyway (mic
and speaker share one I2S port). Cost: stt model time 586 -> ~655 ms on a 4 s clip (its two
staging buffers now share a region; the separate-bank placement of the ESP-IDF app is lost).
Bluetooth is off in this board config: its controller reservation was the internal RAM
the pool needed.

`examples/wake_demo.py`: continuous capture from the ES7210 (machine.I2S, stereo, MCLK),
a 4 s window scored once a second in the background, then the command after the phrase
(silence-trimmed). Needs `mpy/drivers/io_ext.py` / `es7210.py` on the board.

## Measured (int4 model, 4.1 s clip, from MicroPython)

| | |
|---|---|
| model time, no cache | 586 ms (front end 195, decode 1) |
| model time, 6MB cache | 546 ms |
| wake-phrase scoring (512 sequences) | +153 ms |
| internal RAM free | 174 KB at boot, 133 KB after `stt` opens (+32KB staging buffer, task stacks) |
| PSRAM free | 7.98 MB without cache, 1.99 MB with the 6MB cache |
| app image | 1.97 MB (MicroPython + camera + mp_jpeg + stt) |

Background mode under asyncio: a 20 ms asyncio ticker keeps running while a 4 s clip is
transcribed (600 ms model time). A tight `time.sleep_ms(10)` poll loop doubles model time
(short sleeps busy-wait on the same core); poll at >= 20 ms or from asyncio.

## Flash layout (`boards/SENTRY_S3_STT/partitions-stt.csv`)

| offset | size | |
|---|---|---|
| 0x10000 | 4 MB | MicroPython app (3.6 MB with camera + mp_jpeg + stt + tts) |
| 0x410000 | 6 MB | `model` — `data/models/mmrt/citrinet256_int4.mmrt` (5.99 MB) |
| 0xA10000 | 6 MB | `vfs`, created by MicroPython at boot |

An int8 model (9.78 MB) needs a 10 MB model partition and leaves ~3 MB of filesystem.

## Build

```bash
mpy/build.sh            # FW_DIR=~/build/sentry-fw: micropython v1.27.0 (I2S-MCK patch),
                        # micropython-camera-API (fb_location patch), mp_jpeg;
                        # IDF_PATH default ~/build/dstike-fw/esp-idf (v5.5.1)
```

FW_DIR: MicroPython v1.27.0 with `mpy/patches/micropython-i2s-mck.patch` applied (lets
`machine.I2S` output MCLK, which the ES7210/ES8311 codecs need); add cnadler86's
micropython-camera-API and mp_jpeg next to it for the camera module (optional).
`build.sh` adds the board definition (`boards/SENTRY_S3_STT`), `usermods.cmake` (camera +
stt) and `components/` (stt_engine, which pulls esp-dsp through the component manager),
and merges `FW_DIR/out/firmware-stt.bin` with the built flash mode kept (QIO 80 MHz: the
model streams from flash).

## Flash

```bash
esptool.py --chip esp32s3 -p /dev/ttyACM0 -b 460800 write_flash --flash_mode keep \
  --flash_freq keep --flash_size keep \
  0x0 ~/build/sentry-fw/out/firmware-stt.bin 0x410000 data/models/mmrt/citrinet256_int4.mmrt
esptool.py --chip esp32s3 -p /dev/ttyACM0 erase_region 0xA10000 0x5F0000   # first time only
```

Traps hit bringing this up:
- **Erase the vfs region on first flash.** Leftover data there (an old app or model) makes
  MicroPython loop on "The filesystem appears to be corrupted" instead of formatting.
- **REPL on USB-Serial-JTAG, not TinyUSB** (`MICROPY_HW_ENABLE_USBDEV (0)` in the board's
  mpconfigboard.h). With TinyUSB CDC the board changes USB identity (303a:1001 -> 303a:4001),
  and a firmware that fails early vanishes from USB entirely: recovering needed the BOOT
  button. On USB-Serial-JTAG it stays 303a:1001, esptool resets it, boot logs are visible.
- Changing the partition layout from the stock Sentry image (`4MiBplus`: app 2MB, vfs from
  0x200000) wipes the node's filesystem: re-provision after flashing a Sentry.
