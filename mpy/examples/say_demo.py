# Speak text through the Waveshare ESP32-S3-CAM's speaker (ES8311 + PA), all on-device:
# tts (sanoTTS nano voice, 24 kHz) -> machine.I2S (STEREO, MCLK) -> ES8311.
#
#   mpremote cp io_ext.py es8311.py say_demo.py :
#   mpremote exec "import say_demo; say_demo.say('Hello from a tiny computer.')"
import time

import tts
from machine import I2C, I2S, Pin

PINS = dict(scl=7, sda=8, mclk=10, bclk=11, ws=12, dout=14)
_i2c = _io = None


def _audio_on(volume):
    global _i2c, _io
    from es8311 import ES8311
    from io_ext import IOExt
    if _i2c is None:
        _i2c = I2C(0, scl=Pin(PINS["scl"]), sda=Pin(PINS["sda"]), freq=100000)
        _io = IOExt(_i2c, 0x24)
    _io.pin(6, 1)                     # audio rail
    _io.pin(4, 0)                     # PA off until I2S runs
    time.sleep_ms(50)
    dac = ES8311(_i2c, 0x18)
    dac.init(bits=16)
    dac.volume(volume)
    return I2S(0, sck=Pin(PINS["bclk"]), ws=Pin(PINS["ws"]), mck=Pin(PINS["mclk"]), sd=Pin(PINS["dout"]),
               mode=I2S.TX, bits=16, format=I2S.STEREO, rate=tts.RATE, ibuf=32000)


def say(text, volume=85, gain=0.77):     # volume: ES8311 DAC 0-100 (75 = 0 dB); gain: PCM peak
    t0 = time.ticks_ms()
    pcm = tts.say(text, gain, True)                 # stereo int16 @ 24 kHz
    st = tts.last()
    print("synth %d ms for %.2f s of audio (RTF %.2f)" % (st["synth_ms"], st["samples"] / tts.RATE, st["rtf"]))
    i2s = _audio_on(volume)
    _io.pin(4, 1)                                   # PA on
    mv = memoryview(pcm)
    for i in range(0, len(pcm), 4096):
        i2s.write(mv[i:i + 4096])
    time.sleep_ms(300)                              # let the DMA drain
    _io.pin(4, 0)
    i2s.deinit()
    return time.ticks_diff(time.ticks_ms(), t0)
