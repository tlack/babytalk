# Board side of field/capture.py: record the Waveshare ESP32-S3-CAM's two ES7210 mics and
# send the raw PCM to the laptop over a WiFi TCP socket (the USB console only carries
# commands: base64 over it ran at ~8 KB/s). Needs mpy/drivers/io_ext.py + es7210.py.
#
#   setup(gain) -> "READY"; net(ssid, pw, port) -> "LISTEN <ip> <port>"; accept() -> "CONNECTED"
#   record(secs): "REC <secs>" when capture starts (the laptop starts TTS playback on it),
#                 then "DATA <nbytes>" and exactly that many bytes of interleaved stereo int16
#                 LE PCM on the socket, then "END".
import time

from machine import I2C, I2S, Pin

RATE = 16000
PINS = dict(scl=7, sda=8, mclk=10, bclk=11, ws=12, din=13)
CHUNK = 4096
_buf = None
_srv = None
_conn = None


def setup(gain=10, max_secs=26):
    global _buf
    from es7210 import ES7210
    from io_ext import IOExt
    i2c = I2C(0, scl=Pin(PINS["scl"]), sda=Pin(PINS["sda"]), freq=100000)
    io = IOExt(i2c, 0x24)
    io.pin(6, 1)                 # audio rail
    io.pin(4, 0)                 # speaker amp off
    time.sleep_ms(50)
    ES7210(i2c, 0x40).init(rate=RATE, bits=16, gain=gain)
    if _buf is None or len(_buf) < max_secs * RATE * 4:
        _buf = bytearray(max_secs * RATE * 4)
    print("READY gain=%d" % gain)


def net(ssid, pw, port=5556, timeout_s=20):
    """Join WiFi (credentials passed in by the laptop, never stored) and listen on port."""
    global _srv
    import network
    import socket
    w = network.WLAN(network.STA_IF)
    w.active(True)
    if not w.isconnected():
        w.connect(ssid, pw)
        t0 = time.ticks_ms()
        while not w.isconnected():
            if time.ticks_diff(time.ticks_ms(), t0) > timeout_s * 1000:
                raise OSError("wifi: no connection to %r" % ssid)
            time.sleep_ms(200)
    if _srv is None:
        _srv = socket.socket()
        _srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _srv.bind(("0.0.0.0", port))
        _srv.listen(1)
    print("LISTEN %s %d" % (w.ifconfig()[0], port))


def accept():
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn, _ = _srv.accept()
    print("CONNECTED")


def record(secs):
    n = min(len(_buf), int(secs * RATE) * 4)
    mv = memoryview(_buf)
    i2s = I2S(0, sck=Pin(PINS["bclk"]), ws=Pin(PINS["ws"]), mck=Pin(PINS["mclk"]), sd=Pin(PINS["din"]),
              mode=I2S.RX, bits=16, format=I2S.STEREO, rate=RATE, ibuf=8192)
    tmp = bytearray(CHUNK)
    for _ in range(4):           # drop the first ~250 ms: codec/PGA settling
        i2s.readinto(tmp)
    print("REC %.2f" % secs)
    got = 0
    while got < n:
        got += i2s.readinto(mv[got:min(n, got + CHUNK)])
    i2s.deinit()
    print("DATA %d" % n)
    sent = 0
    while sent < n:
        sent += _conn.write(mv[sent:min(n, sent + 8192)])
    print("END")
