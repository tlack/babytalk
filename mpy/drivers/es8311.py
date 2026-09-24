# es8311.py (/lib) — ES8311 mono codec, DAC/speaker path (Waveshare ESP32-S3-CAM).
#
# Port of the vendor/Espressif es8311 driver, minus its ~100-row clock coefficient
# table. Every row of that table where MCLK == 256 * Fs — i.e. 8k/11.025k/12k/16k/
# 22.05k/24k/32k/44.1k/48k/64k — carries the SAME coefficients:
#
#   pre_div=1 pre_multi=0 adc_div=1 dac_div=1 fs_mode=0
#   lrck_h=0x00 lrck_l=0xff bclk_div=4 adc_osr=0x10 dac_osr=0x10
#
# machine.I2S clocks MCLK at 256*Fs natively, so the dividers below are constants and
# the sample rate becomes a property of the I2S peripheral alone — change the rate and
# the codec needs no retuning. Proof + the derivation: BOARD_WAVESHARE_S3_CAM.md §4.
#
# The chip is an I2S SLAVE here: the ESP32 drives MCLK/BCLK/LRCK.

import time

# --- registers ---
REG_RESET = 0x00
REG_CLK_MANAGER = 0x01
REG_CLK_DIV = 0x02       # pre_div / pre_multi
REG_ADC_OSR = 0x03       # fs_mode | adc_osr
REG_DAC_OSR = 0x04
REG_ADC_DAC_DIV = 0x05
REG_BCLK_DIV = 0x06
REG_LRCK_H = 0x07
REG_LRCK_L = 0x08
REG_SDPIN = 0x09
REG_SDPOUT = 0x0A
REG_SYSTEM_0D = 0x0D
REG_SYSTEM_0E = 0x0E
REG_SYSTEM_12 = 0x12
REG_SYSTEM_13 = 0x13
REG_SYSTEM_14 = 0x14     # DMIC select + analog PGA gain
REG_ADC_17 = 0x17        # ADC volume
REG_ADC_1C = 0x1C        # ADC equalizer
REG_DAC_VOLUME = 0x32
REG_DAC_37 = 0x37        # ramp rate
REG_CHIP_ID1 = 0xFD
REG_CHIP_ID2 = 0xFE

# resolution -> SDP register nibble (both SDP in and out use the same encoding)
_RES = {16: 0x0C, 18: 0x08, 20: 0x04, 24: 0x00, 32: 0x10}


class ES8311:
    def __init__(self, i2c, addr=0x18):
        self.i2c = i2c
        self.addr = addr
        self._volume = 0

    # --- register helpers ---
    def _w(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes((val & 0xFF,)))

    def _r(self, reg):
        return self.i2c.readfrom_mem(self.addr, reg, 1)[0]

    def chip_id(self):
        """(0x83, 0x11) on a healthy ES8311 — a cheap 'is it really there' probe."""
        return self._r(REG_CHIP_ID1), self._r(REG_CHIP_ID2)

    # --- bring-up ---
    def init(self, bits=16):
        """Reset, clock, format and power up the DAC path. MCLK must be 256*Fs."""
        res = _RES.get(bits)
        if res is None:
            raise ValueError("es8311: unsupported resolution %r" % bits)

        # reset to defaults, then power on
        self._w(REG_RESET, 0x1F)
        time.sleep_ms(20)
        self._w(REG_RESET, 0x00)
        self._w(REG_RESET, 0x80)

        # clock source = MCLK pin, all clocks enabled
        self._w(REG_CLK_MANAGER, 0x3F)

        # dividers — the fixed 256*Fs row (see module docstring)
        self._w(REG_CLK_DIV, self._r(REG_CLK_DIV) & 0x07)      # pre_div=1, pre_multi=0
        self._w(REG_ADC_OSR, 0x10)                              # fs_mode=0 | adc_osr
        self._w(REG_DAC_OSR, 0x10)
        self._w(REG_ADC_DAC_DIV, 0x00)                          # adc_div=1, dac_div=1
        self._w(REG_BCLK_DIV, (self._r(REG_BCLK_DIV) & 0xE0) | 0x03)   # bclk_div=4
        self._w(REG_LRCK_H, self._r(REG_LRCK_H) & 0xC0)         # lrck_h=0x00
        self._w(REG_LRCK_L, 0xFF)

        # slave mode + I2S format at the requested resolution
        self._w(REG_RESET, self._r(REG_RESET) & 0xBF)
        self._w(REG_SDPIN, res)
        self._w(REG_SDPOUT, res)

        # power up analog, DAC, and the output driver
        self._w(REG_SYSTEM_0D, 0x01)
        self._w(REG_SYSTEM_0E, 0x02)
        self._w(REG_SYSTEM_12, 0x00)
        self._w(REG_SYSTEM_13, 0x10)
        self._w(REG_ADC_1C, 0x6A)
        self._w(REG_DAC_37, 0x08)

    def volume(self, pct=None):
        """DAC volume as 0-100. Returns the current setting when called with no arg."""
        if pct is None:
            return self._volume
        pct = 0 if pct < 0 else (100 if pct > 100 else pct)
        self._w(REG_DAC_VOLUME, 0 if pct == 0 else (pct * 256 // 100) - 1)
        self._volume = pct
        return pct

    def microphone(self, digital=False):
        """Configure the codec's OWN mic input. Unused on this board — the mic array is
        an ES7210 on the same I2S bus — but kept because it costs two writes and the
        next ES8311 board may well use it."""
        self._w(REG_ADC_17, 0xC8)
        self._w(REG_SYSTEM_14, 0x1A | (0x40 if digital else 0x00))

    def mute(self, on=True):
        self._w(REG_SYSTEM_12, 0x02 if on else 0x00)
