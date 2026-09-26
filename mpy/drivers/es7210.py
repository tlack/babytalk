# es7210.py (/lib) — ES7210 4-channel mic ADC, capture path (Waveshare ESP32-S3-CAM).
#
# Port of the vendor/Espressif es7210 driver. Like es8311.py this drops the clock
# coefficient table: for MCLK == 256 * Fs at 16k / 44.1k / 48k the rows are identical
# (adc_div=1, dll=1, doubler=1, osr=0x20, lrck=0x0100), and lrck is just mclk/rate, so
# the whole table reduces to the constants below. 64k differs only in `doubler`, which
# is why RATES is an explicit whitelist rather than an open formula — an unlisted rate
# would silently get wrong dividers, and silence is the one failure mode that looks
# exactly like working hardware.
#
# The chip is an I2S SLAVE; the ESP32 drives MCLK/BCLK/LRCK. The board wires two mics
# of the four channels, so a stereo 2-slot read gives both; we mix or take one.
#
# Derived from Espressif's es7210 driver (github.com/espressif/esp-bsp, components/es7210),
# Copyright Espressif Systems (Shanghai) CO LTD, Apache-2.0 (licenses/Apache-2.0.txt).
# Changes: rewritten in MicroPython, clock table reduced to its MCLK = 256 x Fs rows.

import time

# --- registers ---
REG_RESET = 0x00
REG_MAINCLK = 0x02       # adc_div | doubler<<6 | dll<<7
REG_LRCK_DIVH = 0x04
REG_LRCK_DIVL = 0x05
REG_POWER_DOWN = 0x06
REG_OSR = 0x07
REG_TIME_CTRL0 = 0x09
REG_TIME_CTRL1 = 0x0A
REG_SDP_IF1 = 0x11       # fmt | bit width
REG_SDP_IF2 = 0x12       # TDM
REG_ADC34_HPF2 = 0x20
REG_ADC34_HPF1 = 0x21
REG_ADC12_HPF2 = 0x22
REG_ADC12_HPF1 = 0x23
REG_ADC1_DB = 0x1B
REG_ADC2_DB = 0x1C
REG_ADC3_DB = 0x1D
REG_ADC4_DB = 0x1E
REG_ANALOG = 0x40
REG_MIC12_BIAS = 0x41
REG_MIC34_BIAS = 0x42
REG_MIC1_GAIN = 0x43
REG_MIC2_GAIN = 0x44
REG_MIC3_GAIN = 0x45
REG_MIC4_GAIN = 0x46
REG_MIC1_POWER = 0x47
REG_MIC2_POWER = 0x48
REG_MIC3_POWER = 0x49
REG_MIC4_POWER = 0x4A
REG_MIC12_POWER = 0x4B
REG_MIC34_POWER = 0x4C

# bit width -> REG_SDP_IF1 high nibble
_BITS = {16: 0x60, 18: 0x40, 20: 0x20, 24: 0x00, 32: 0x80}

# rates verified against the vendor coefficient table at MCLK = 256*Fs
RATES = (16000, 44100, 48000)

MIC_BIAS_2V87 = 0x70     # vendor default for this board's analog mics
GAIN_MAX = 0x0E          # 0x00..0x0E, ~3dB per step


class ES7210:
    def __init__(self, i2c, addr=0x40):
        self.i2c = i2c
        self.addr = addr

    def _w(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes((val & 0xFF,)))

    def _r(self, reg):
        return self.i2c.readfrom_mem(self.addr, reg, 1)[0]

    def init(self, rate=16000, bits=16, gain=0x08, mic_bias=MIC_BIAS_2V87):
        """Reset and configure the ADC for I2S slave capture at `rate`."""
        if rate not in RATES:
            raise ValueError("es7210: rate %r not in %r" % (rate, RATES))
        bw = _BITS.get(bits)
        if bw is None:
            raise ValueError("es7210: unsupported resolution %r" % bits)

        # software reset
        self._w(REG_RESET, 0xFF)
        self._w(REG_RESET, 0x32)

        # power-up timing
        self._w(REG_TIME_CTRL0, 0x30)
        self._w(REG_TIME_CTRL1, 0x30)

        # high-pass filters on all four ADCs (kills DC offset / rumble)
        self._w(REG_ADC12_HPF1, 0x2A)
        self._w(REG_ADC12_HPF2, 0x0A)
        self._w(REG_ADC34_HPF1, 0x2A)
        self._w(REG_ADC34_HPF2, 0x0A)

        # standard I2S, no TDM
        self._w(REG_SDP_IF1, 0x00 | bw)
        self._w(REG_SDP_IF2, 0x00)

        # analog power + VMID
        self._w(REG_ANALOG, 0xC3)
        self._w(REG_MIC12_BIAS, mic_bias)
        self._w(REG_MIC34_BIAS, mic_bias)
        self.gain(gain)

        # power on the mic front-ends
        for r in (REG_MIC1_POWER, REG_MIC2_POWER, REG_MIC3_POWER, REG_MIC4_POWER):
            self._w(r, 0x08)

        # clocking — the fixed 256*Fs row (see module docstring)
        lrck = 256                      # mclk / rate when mclk = 256*rate
        self._w(REG_OSR, 0x20)
        self._w(REG_MAINCLK, 0x01 | (0x01 << 6) | (0x01 << 7))   # adc_div=1 doubler dll
        self._w(REG_LRCK_DIVH, (lrck >> 8) & 0x0F)
        self._w(REG_LRCK_DIVL, lrck & 0xFF)

        # DLL down, bias/ADC/PGA up, then enable
        self._w(REG_POWER_DOWN, 0x04)
        self._w(REG_MIC12_POWER, 0x0F)
        self._w(REG_MIC34_POWER, 0x0F)
        self._w(REG_RESET, 0x71)
        self._w(REG_RESET, 0x41)

    def gain(self, g):
        """Analog PGA gain, 0x00-0x0E (~3dB/step)."""
        g = 0 if g < 0 else (GAIN_MAX if g > GAIN_MAX else g)
        for r in (REG_MIC1_GAIN, REG_MIC2_GAIN, REG_MIC3_GAIN, REG_MIC4_GAIN):
            self._w(r, g | 0x10)
        return g

    def volume_db(self, db):
        """Digital gain, -95..+32 dB in 0.5dB steps (0xBF == 0dB)."""
        db = -95 if db < -95 else (32 if db > 32 else db)
        v = 191 + int(db * 2)
        for r in (REG_ADC1_DB, REG_ADC2_DB, REG_ADC3_DB, REG_ADC4_DB):
            self._w(r, v)
        return db
