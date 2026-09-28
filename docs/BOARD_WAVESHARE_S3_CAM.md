# Waveshare ESP32-S3-CAM: the audio facts BabyTalk relies on

Product: <https://www.waveshare.com/esp32-s3-cam-ov5640.htm> ·
Docs: <https://docs.waveshare.com/ESP32-S3-CAM-OVxxxx> ·
Vendor code (BSP, pin header, examples): <https://github.com/waveshareteam/ESP32-S3-CAM-OVxxxx>

The drivers that use this: `mpy/drivers/` (MicroPython), `stt/main/mic.c`,
`bench/main/bench_rec.c` and `atomvm/components/atomvm_babytalk/board_audio.c` (AtomVM, where
these pins are the preset `waveshare_s3_cam` of `babytalk:audio_config/1`).

## 1. What's on it

| Block | Part | Notes |
|---|---|---|
| SoC | ESP32-S3R8 | 16 MB flash, **8 MB octal PSRAM** (`CONFIG_SPIRAM_MODE_OCT=y`) |
| Mic | **ES7210** 4-channel ADC, dual-mic array | I2S in; mic 1 on the left slot, mic 2 on the right |
| Speaker | **ES8311** mono codec -> **NS4150B** 3 W class-D amp | speaker on an MX1.25 2-pin header |
| IO expander | **CH32V003** (a small RISC-V MCU) | gates the audio rail and the amp -- see 3.2 |
| Camera | OV5640 (socketed DVP) | unused by BabyTalk; shares the I2C bus |

## 2. Pins (from the vendor BSP header `esp32_s3_cam_ovxxxx.h`)

| Signal | GPIO / I2C address |
|---|---|
| I2C SCL / SDA (one bus for everything) | 7 / 8 |
| ES8311 | `0x18` |
| ES7210 | `0x40` |
| CH32V003 expander | `0x24` |
| I2S MCLK / BCLK / LRCK (WS) | 10 / 11 / 12 |
| I2S DIN (mic -> ESP32) | 13 |
| I2S DOUT (ESP32 -> speaker) | 14 |

Both codecs share one I2S port's clock pins, so capture and playback take turns.

## 3. Traps

### 3.1 One I2C bus
The camera's SCCB is the same bus (7/8) as both codecs and the expander. Create **one** I2C
master and share it; a second driver on the same pins fails with timeouts.

### 3.2 The amp and the audio rail are expander pins, not GPIOs
The vendor BSP's amp-enable pin is "not connected" because it lives on the CH32V003:
**P6 = audio rail**, **P4 = NS4150B amp enable** (confirmed by ear, 2026-09-21; inferred from
the order of the vendor audio example, not from its pin comments, which are copied from
another board). The expander's output register (`0x03`) is write-only and takes all eight
pins at once, so keep a shadow copy; `0x02` sets the direction (1 = output). Raise P6 before
initialising the codecs; raise P4 only once I2S is streaming (no turn-on pop), and drop it
when done.

### 3.3 I2S MONO garbles this codec pair: always run STEREO
MONO leaves one slot empty and audibly modulates speech; a sine test hides it completely.
Capture STEREO and keep the left slot; play mono as the same sample in both slots.

## 4. ES8311 clocks: the coefficient table collapses

The vendor driver carries ~100 `{mclk, rate, ...}` divider rows. Every row where
**MCLK = 256 x Fs** (8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48, 64 kHz) holds the same set:

```
pre_div=1  pre_multi=0  adc_div=1  dac_div=1  fs_mode=0
lrck_h=0x00  lrck_l=0xff  bclk_div=4  adc_osr=0x10  dac_osr=0x10
```

With MCLK always at 256 x Fs (ESP-IDF's `I2S_STD_CLK_DEFAULT_CONFIG` and the patched
`machine.I2S` both do this), the divider writes are constants:

| reg | value |
|---|---|
| `0x02` | `(old & 0x07)` |
| `0x03` | `0x10` |
| `0x04` | `0x10` |
| `0x05` | `0x00` |
| `0x06` | `(old & 0xE0) \| 0x03` |
| `0x07` | `(old & 0xC0)` |
| `0x08` | `0xff` |

The sample rate is then set by the I2S clock alone: the codec needs no re-tuning to switch
between 16 kHz capture and 24 kHz speech. The ES7210 reduces the same way (its rows at
16/44.1/48 kHz are identical; see `mpy/drivers/es7210.py`).

## 5. DAC volume

ES8311 register `0x32`: `0xBF` = 0 dB, 0.5 dB per step. BabyTalk's volume V (0..100) writes
`V * 256 / 100 - 1`, so 75 = 0 dB. Speech already peaks at 77% of full scale; much above
~80 clips in the DAC (85 = +12.5 dB sounded saturated).
