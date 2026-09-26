# io_ext.py (/lib) — CH32V003 I2C IO expander (Waveshare ESP32-S3-CAM).
#
# A tiny RISC-V MCU acting as an 8-bit IO expander + 1 PWM + 1 ADC, sitting at 0x24 on
# the shared system I2C bus. It matters to us because the NS4150B amplifier enable and
# the audio rail are expander pins, NOT GPIOs — docs/BOARD_WAVESHARE_S3_CAM.md section 3.2.
#
# Register map (vendor io_extension.c):
#   0x02 <mask>  IO direction, 1 = output
#   0x03 <byte>  output latch (all 8 pins at once — we shadow it, see below)
#   0x04 ->      input byte
#   0x05 <duty>  PWM (0-255), the LCD backlight on boards that have one
#   0x06 ->      ADC, 16-bit little-endian
#
# The output register is write-only and takes the WHOLE byte, so a read-modify-write is
# impossible and the shadow copy is the only truth. Keep one instance per bus.

_REG_MODE = 0x02
_REG_OUT = 0x03
_REG_IN = 0x04
_REG_PWM = 0x05
_REG_ADC = 0x06


class IOExt:
    def __init__(self, i2c, addr=0x24, out_mask=0xFF, initial=0x00):
        self.i2c = i2c
        self.addr = addr
        self._shadow = initial & 0xFF
        # All pins output by default; the board's audio pins are all outputs.
        self.i2c.writeto(addr, bytes((_REG_MODE, out_mask & 0xFF)))
        self.i2c.writeto(addr, bytes((_REG_OUT, self._shadow)))

    def pin(self, n, value):
        """Drive expander pin n (0-7) high or low."""
        if value:
            self._shadow |= 1 << n
        else:
            self._shadow &= ~(1 << n) & 0xFF
        self.i2c.writeto(self.addr, bytes((_REG_OUT, self._shadow)))

    def get(self, n):
        """Read expander pin n (0-7)."""
        self.i2c.writeto(self.addr, bytes((_REG_IN,)))
        v = self.i2c.readfrom(self.addr, 1)[0]
        return (v >> n) & 1

    def pwm(self, duty):
        """PWM output, 0-255. (Backlight on display-fitted variants; unused here.)"""
        self.i2c.writeto(self.addr, bytes((_REG_PWM, duty & 0xFF)))

    def adc(self):
        """16-bit ADC reading (battery sense on variants that wire it)."""
        self.i2c.writeto(self.addr, bytes((_REG_ADC,)))
        d = self.i2c.readfrom(self.addr, 2)
        return d[0] | (d[1] << 8)

    @property
    def shadow(self):
        return self._shadow
