# Wake phrase + command, all on the board, from MicroPython (firmware with the `stt` module).
#
#   mpremote cp io_ext.py es7210.py wake_up_tomato_face.json wake_demo.py :
#   mpremote exec "import wake_demo; wake_demo.run()"
#
# io_ext.py / es7210.py: the watchtower node drivers (audio rail on the IO expander, the
# ES7210 mic ADC). The phrase file comes from export/kws.py --save (tools/wake_phrases/).
# Board: Waveshare ESP32-S3-CAM. Capture is STEREO with MCLK (MONO garbles these codecs;
# MCLK needs the patched machine.I2S), and stt takes the left slot itself (channels=2).
import asyncio
import json
import time

import stt
from machine import I2C, I2S, Pin

RATE = 16000
FRAME = 4                   # bytes per stereo 16-bit frame
WIN_S = 4                   # scored window
EVERY_MS = 1000             # how often the window is scored
CMD_S = 3                   # command audio after the phrase
CHUNK = 4096                # bytes per I2S read (64 ms)
PINS = dict(scl=7, sda=8, mclk=10, bclk=11, ws=12, din=13)


def audio_on():
    from es7210 import ES7210
    from io_ext import IOExt
    i2c = I2C(0, scl=Pin(PINS["scl"]), sda=Pin(PINS["sda"]), freq=100000)
    io = IOExt(i2c, 0x24)
    io.pin(6, 1)            # audio rail
    io.pin(4, 0)            # speaker amp off
    time.sleep_ms(50)
    ES7210(i2c, 0x40).init(rate=RATE, bits=16, gain=0x08)
    return I2S(0, sck=Pin(PINS["bclk"]), ws=Pin(PINS["ws"]), mck=Pin(PINS["mclk"]), sd=Pin(PINS["din"]),
               mode=I2S.RX, bits=16, format=I2S.STEREO, rate=RATE, ibuf=RATE * FRAME)


class Window:
    """The latest `secs` of stereo audio, oldest first (a shift register: stt wants it
    contiguous, and a 256KB memmove per 64 ms chunk is cheap next to the model)."""

    def __init__(self, secs):
        self.buf = bytearray(secs * RATE * FRAME)
        self.mv = memoryview(self.buf)
        self.chunk = bytearray(CHUNK)

    def pump(self, i2s):
        n = i2s.readinto(self.chunk)
        if n:
            self.mv[:-n] = self.mv[n:]
            self.mv[-n:] = memoryview(self.chunk)[:n]
        return n

    def clear(self):
        for i in range(0, len(self.buf), CHUNK):
            self.mv[i:i + CHUNK] = bytes(min(CHUNK, len(self.buf) - i))


def trim_silence(pcm, step=CHUNK):
    """Drop trailing chunks quieter than 2x the quietest one (the room): the model
    otherwise 'hears' words in fan noise after the command."""
    mv = memoryview(pcm)
    levels = [stt.rms(mv[i:i + step], 2) for i in range(0, len(pcm) - step + 1, step)]
    if not levels:
        return pcm
    floor = max(30, min(levels))
    end = len(levels)
    while end > 1 and levels[end - 1] < 2 * floor:
        end -= 1
    return mv[:min(len(pcm), (end + 3) * step)]     # keep ~200 ms of tail


async def transcribe(pcm, channels=2):
    stt.start(pcm, channels)          # background task; poll >= 20 ms (see mpy/README.md)
    while stt.busy():
        await asyncio.sleep_ms(20)
    return stt.result(), stt.last()


async def main(phrase_file, threshold):
    cfg = json.load(open(phrase_file))
    thr = cfg["threshold"] if threshold is None else threshold
    print("phrase", cfg["phrase"], "-", stt.phrase(cfg["spellings"]), "sequences, threshold", thr)
    i2s = audio_on()
    win = Window(WIN_S)
    pending = None
    since = 0                                        # bytes captured after the snapshot
    last = time.ticks_ms()
    print('say "%s" ...' % cfg["phrase"])
    while True:
        since += win.pump(i2s)
        await asyncio.sleep_ms(0)
        if pending is None and time.ticks_diff(time.ticks_ms(), last) >= EVERY_MS:
            last = time.ticks_ms()
            stt.start(win.buf, 2)
            pending = bytes(win.buf)                 # the audio that was scored
            since = 0
        if pending is not None and not stt.busy():
            text = stt.result()
            d = stt.last()
            snap, pending = pending, None
            if d["score"] is not None and d["score"] >= thr:
                print("\nAWAKE  score %.1f  (heard %r, %d ms)" % (d["score"], text, d["model_ms"]))
                # the command: audio after the phrase in the scored window, what arrived
                # while it was being scored, then CMD_S more seconds
                cut = min(len(snap), (d["span"][1] + 1) * 8 * 160 * FRAME)
                cmd = bytearray(snap[cut:])
                if since:
                    cmd += win.mv[-min(since, len(win.buf)):]
                t_end = time.ticks_add(time.ticks_ms(), CMD_S * 1000)
                chunk = bytearray(CHUNK)
                while time.ticks_diff(t_end, time.ticks_ms()) > 0:
                    n = i2s.readinto(chunk)
                    cmd += chunk[:n]
                    await asyncio.sleep_ms(0)
                ctext, cd = await transcribe(trim_silence(cmd))
                print("COMMAND %r  (%d ms)\n" % (ctext, cd["model_ms"]))
                win.clear()
                last = time.ticks_ms()


def run(phrase_file="wake_up_tomato_face.json", threshold=None):
    try:
        asyncio.run(main(phrase_file, threshold))
    except KeyboardInterrupt:
        print("stopped")
