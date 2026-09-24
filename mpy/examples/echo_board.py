# Echo: record from the board's mic, transcribe on-device (stt), speak it back (tts).
# Board side of tools/echo.py; also usable alone:
#   import echo_board; echo_board.setup(); echo_board.take(4)
# Needs io_ext.py, es7210.py, es8311.py (watchtower node drivers) on the board.
import time

import stt
import tts
from machine import I2C, I2S, Pin

RATE = 16000                 # mic rate (the stt model's)
PINS = dict(scl=7, sda=8, mclk=10, bclk=11, ws=12, din=13, dout=14)
CHUNK = 4096                 # bytes per I2S read, stereo: 64 ms
_i2c = _io = _buf = None


def setup(max_secs=8):
    """Reserve the shared SRAM pool, power the audio rail, init both codecs."""
    global _i2c, _io, _buf
    tts.reserve()
    from es7210 import ES7210
    from io_ext import IOExt
    _i2c = I2C(0, scl=Pin(PINS["scl"]), sda=Pin(PINS["sda"]), freq=100000)
    _io = IOExt(_i2c, 0x24)
    _io.pin(6, 1)            # audio rail
    _io.pin(4, 0)            # speaker amp off
    time.sleep_ms(50)
    ES7210(_i2c, 0x40).init(rate=RATE, bits=16, gain=0x08)
    _buf = bytearray(max_secs * RATE * 4)
    print("READY")


def _i2s(mode, rate):
    rx = mode == "rx"
    return I2S(0, sck=Pin(PINS["bclk"]), ws=Pin(PINS["ws"]), mck=Pin(PINS["mclk"]),
               sd=Pin(PINS["din"] if rx else PINS["dout"]), mode=I2S.RX if rx else I2S.TX,
               bits=16, format=I2S.STEREO, rate=rate, ibuf=8192)  # DMA: internal RAM, small


def _record(secs):
    n = min(len(_buf), int(secs * RATE) * 4)
    mv = memoryview(_buf)
    i2s = _i2s("rx", RATE)
    got = 0
    while got < n:
        got += i2s.readinto(mv[got:min(n, got + CHUNK)])
    i2s.deinit()
    return mv[:n]


def _trim(pcm, step=CHUNK):
    """Keep from ~200 ms before the first loud chunk to ~200 ms after the last one
    (loud = over 2x the quietest chunk): the model otherwise 'hears' words in room noise."""
    lv = [stt.rms(pcm[i:i + step], 2) for i in range(0, len(pcm) - step + 1, step)]
    if not lv:
        return pcm
    floor = max(30, min(lv))
    loud = [i for i, v in enumerate(lv) if v >= 2 * floor]
    if not loud:
        return None
    a, b = max(0, loud[0] - 3), min(len(lv), loud[-1] + 4)
    return pcm[a * step:b * step]


def _speak(text, volume=85, peak=0.77):
    pcm = tts.say(text, peak, True)
    st = tts.last()
    from es8311 import ES8311
    dac = ES8311(_i2c, 0x18)
    dac.init(bits=16)
    dac.volume(volume)
    i2s = _i2s("tx", tts.RATE)
    _io.pin(4, 1)
    mv = memoryview(pcm)
    for i in range(0, len(pcm), 4096):
        i2s.write(mv[i:i + 4096])
    time.sleep_ms(300)
    _io.pin(4, 0)
    i2s.deinit()
    return st


def take(secs=4, prefix="You said: "):
    print("REC %.1f" % secs)
    pcm = _record(secs)
    print("PROCESSING")
    speech = _trim(pcm)
    text = stt.transcribe(speech, 2) if speech is not None and len(speech) >= 4 * 1600 else ""
    d = stt.last() if text else None
    print("HEARD %s" % text)
    if d:
        print("STT %.0f ms for %.1f s" % (d["fe_ms"] + d["model_ms"], len(speech) / 4 / RATE))
    st = _speak(prefix + text if text else "Sorry, I didn't catch that.")
    print("TTS %.0f ms for %.1f s (RTF %.2f)" % (st["synth_ms"], st["samples"] / tts.RATE, st["rtf"]))
    print("DONE")
