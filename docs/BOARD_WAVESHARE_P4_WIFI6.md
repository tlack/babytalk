# Waveshare ESP32-P4-WIFI6: BabyTalk on the ESP32-P4

Product: <https://www.waveshare.com/esp32-p4-wifi6.htm> ·
Wiki (pins, demos): <https://www.waveshare.com/wiki/ESP32-P4-WIFI6>

BabyTalk runs here from **AtomVM** (Erlang / Elixir): speech to text, text to speech,
microphone and speaker. The MicroPython firmware (`mpy/`) is still ESP32-S3 only.

The code that makes it work:

- `mmrt/mmrt_port.c`: MMRT's portable kernels (plain C, bit-exact with `mmrt_ref.c`), used on
  any chip without the S3's vector unit;
- `mmrt/mmrt_p4.S`: the ESP32-P4's vector unit, for the 1x1 and depthwise convolutions;
- `atomvm/components/atomvm_babytalk/board_audio_p4_wifi6.c`: this board's audio chips.

## 1. What's on it

| Block | Part | Notes |
|---|---|---|
| SoC | **ESP32-P4NRW32X** | two 360 MHz RISC-V cores, **32 MB PSRAM** in the package, 32 MB flash. Ours is revision 1.3, which ESP-IDF treats as pre-production silicon (below v3.0) |
| WiFi / Bluetooth | **ESP32-C6** co-processor | the P4 has no radio; the C6 talks to it over SDIO (esp_hosted). **2.4 GHz only** |
| Mic + speaker codec | **ES8311** | one mono codec for both directions: its ADC takes the board's mic, its DAC the speaker |
| Speaker amp | **NS4150B** | 3 W class-D, the speaker on an MX1.25 2-pin header |
| Also | MIPI CSI and DSI connectors, TF slot, USB OTG | unused by BabyTalk |

## 2. Pins (Waveshare's wiki)

| Signal | GPIO / I2C address |
|---|---|
| I2C SDA / SCL | 7 / 8 |
| ES8311 | `0x18` |
| I2S MCLK / BCLK / LRCK (WS) | 13 / 12 / 10 |
| I2S: codec ADC out -> P4 (ASDOUT) | 11 |
| I2S: P4 -> codec DAC in (DSDIN) | 9 |
| NS4150B enable (PA) | 53, high = on |
| Console UART (the USB-C port's CH343) | 37 / 38 |

`board_audio_p4_wifi6.c` is the ES8311 set up for both directions at once (Espressif's
esp_codec_dev register sequence), on one I2S port that the mic and the speaker take turns
on, as on every BabyTalk board. The amp is on only while playing.

## 3. Building for the P4

`atomvm/build.sh` builds for the ESP32-S3. A P4 firmware takes the same components with
these differences:

1. **Target** `esp32p4`, and AtomVM's own preset for a P4 with a C6 for WiFi:
   `sdkconfig.defaults.esp32p4_pre_c6` (revision < 3 silicon) or `..._c6`, copied over
   `sdkconfig.defaults.esp32p4`. AtomVM's CI does the same.
2. **WiFi through the C6**: `components/avm_builtins/idf_component.yml.esp32p4_wifi_remote`
   copied to `idf_component.yml`, which brings `esp_wifi_remote` and `esp_hosted`. AtomVM's
   network module then works unchanged. (Our C6 shipped with esp_hosted firmware older than
   the host's 3.0.9; it runs in a compatibility mode.)
3. AtomVM release-0.7 needs one patch on the P4: its GPIO deep-sleep hold NIFs call
   functions ESP-IDF only declares on chips that have deep-sleep pad hold, and the P4 has none.
   The guard needs `SOC_GPIO_SUPPORT_HOLD_IO_IN_DSLP && !SOC_GPIO_SUPPORT_HOLD_SINGLE_IO_IN_DSLP`.
4. **BabyTalk's settings**: the board `CONFIG_BABYTALK_BOARD_WAVESHARE_P4_WIFI6=y` (the
   default on a P4), and the engines' 84.5 KB pool in PSRAM, `CONFIG_SRAM_POOL_IN_PSRAM=y`.
   On the P4 only speech synthesis uses the pool, and it runs as fast from PSRAM. The internal
   RAM goes to the recognizer's activations instead. Also set `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`
   (sanoTTS's static buffers) and `CONFIG_FREERTOS_TASK_CREATE_ALLOW_EXT_MEM=y`.
5. `atomvm_mmrt` (the `mmrt` module: matvec, matmul... over binaries) works on the P4 too, on
   its vector unit: 256x256 int8 matvec 267 MMAC/s, matmul with 16 rows 1640 MMAC/s, int4 564
   MMAC/s (the S3: 145, 720 and 466). `apps/mmrt_test` passes all 26 test vectors.
6. The P4's bootloader goes at **0x2000** (the S3's at 0x0). With 32 MB of flash, use the S3
   layout or a bigger one; the model partition is found by name (`model`).

## 4. How fast (AtomVM, int4 model)

Measured with `apps/speed_bench`, PSRAM at **200 MHz** (see below):

| task | ESP32-P4 | ESP32-S3 (for comparison) |
|---|---|---|
| transcribe 1 s | 0.20 s | 0.35 s |
| transcribe 2 s | **0.30 s** (the first run after a boot: 0.65 s) | 0.53 s |
| transcribe 4 s | 0.54 s | ~0.85 s |
| transcribe 10 s | 1.35 s | 1.72 s |
| speak 5.4 s of speech | **1.24 s (0.23 x real time)** | 0.27 x real time |

**Set the PSRAM to 200 MHz.** ESP-IDF 5.5 defaults the P4's PSRAM to 20 MHz, and files 200 MHz
under experimental features: `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` and `CONFIG_SPIRAM_SPEED_200M=y`.
At 20 MHz the same firmware took 0.58 s for 2 s of speech; the model's weights and activations
live in PSRAM.

- Transcription on the P4 uses both cores and its vector unit (`mmrt/README.md`, "Other chips:
  the ESP32-P4"). The first transcription after a boot also decodes the int4 weights (9.8 MB,
  kept in PSRAM for later runs).
- Speech synthesis: BabyTalk's own P4 int8 kernels for sanoTTS (`components/sanotts/
  snt_kernels_esp32p4.c`, `snt_matvec_esp32p4.S`) in place of its scalar ones: 0.73x -> 0.23x
  real time, the audio bit-identical. The synthesis `Info` says how much ran on the vector
  unit (`simd_pct`: 100 here). What's left is mostly float work (the spectrum and the inverse
  FFT) and sanoTTS's int16 x int8 kernels, scalar on every chip.
- Memory: 295 KB of internal RAM free at boot, ~227 KB with WiFi and TLS up. PSRAM: 24 MB free
  once the model's rows are unpacked.

**Checked so far.** The whole model, run along the P4's code path on the PC (the vector
dot products in C), is bit-exact with the S3's reference on every one of its 347 tensors. On
the board, each vector kernel checks itself against the portable one before its first use.
The speaker, the mic and speech synthesis work. **Not checked yet**: a transcript of real
speech on this board. A clip played from a nearby laptop speaker barely registered on the
board's mic.

## 5. What the P4's vector unit taught us

The P4's vector instructions (the `Xesppie` RISC-V extension, `esp.` mnemonics) mirror the
S3's (`ee.`), but not exactly. Each of these cost a debugging round:

- **`esp.srcmb.s8.qacc` saturates only when asked.** Its last operand is 1 for "shift and
  clamp to int8". With 0, the S3's value, it keeps the low byte: 3636 became 52, not 127.
- **The S3's rounding trick doesn't carry over.** On the S3, MMRT rounds half up by shifting
  the accumulator in place by one less, adding 1 and shifting by one. On the P4, add
  2^(shift-1) to every lane first (one `esp.vsmulas.s8.qacc` of two int8 factors), then one
  saturating shift.
- **The zero-overhead loop (`esp.lp.setup`) gave wrong sums** when re-armed with a different
  count: a depthwise frame after one clipped at the edge. Plain branch loops work, and cost
  a few percent.
- **`esp.movi.32.q` isn't accepted by the assembler** in any operand order. Load constants
  from memory instead.
- **Only some registers** work as vector-instruction operands: `t0`-`t2` and `a6`/`a7` are
  rejected (x8-x15 and x28-x31 have worked). Assemble a probe before designing around one.
- **`esp.vld.l.64` ignores misalignment silently**: from an address that isn't 8-byte aligned
  it loads the wrong bytes, with no fault. (The int4 decoder copies such rows to an aligned
  buffer first; a test vector with an odd-aligned matrix caught it, the self-check didn't, so
  the self-check now tries unaligned sources too.)
- **The vector unit can't read the P4's low-power RTC RAM**, which ESP-IDF adds to the
  internal heap (a load access fault). Ask for `MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA`, which
  leaves it out.
- **Loop order decides the speed.** One output at a time over every frame is fast while the
  frames fit in cache. Once a clip's activations outgrow it (about 3 s), every frame came from
  PSRAM once per output: 8 s of speech took 30 s. Tiling the frames 32 at a time made it
  3.6 s. Measure several clip lengths, not one.

A bug of our own is worth telling too. For a while every depthwise layer ran twice, once
on the vector unit and then again on the slow reference kernel. A successful call returned 0,
and the dispatch took 0 for "declined". The results were identical, so only the timings
showed it. We then drew a wrong conclusion from those timings ("PSRAM is 50x slower for the
vector unit") until a kernel benchmark in isolation said otherwise.
