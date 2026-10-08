"""A board that streams its microphone over the USB console nonstop (atomvm/apps/mic_stream),
read in the background, with the same record() interface capture.py uses for the MicroPython
board. No WiFi and no commands: the laptop starts its own playback and cuts each clip out of
the stream, then align() finds where the played audio starts in it.

Line protocol (mic_stream.erl):
    BTMIC START <rate> <chunk_ms> <board>
    A <seq> <base64 of 16 kHz mono int16 LE>
    BTMIC STAT <seq> <rms> <psram_free>
    BTMIC ERROR <what>
"""
import base64
import collections
import multiprocessing as mp
import os
import queue
import signal
import threading
import time

import numpy as np
import serial
import soundfile as sf

RATE = 16000
KEEP_SECS = 180        # stream history kept in memory


def _pump(port, q, stop):
    """A child process that only reads the port and stamps each line with its arrival time.
    It has its own interpreter: work in the main one (loading TTS voices holds the GIL for
    seconds) can't keep it from reading, and the board drops output nobody reads."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C is the parent's to handle
    parent = os.getppid()
    ser = None
    while not stop.is_set() and os.getppid() == parent:
        try:
            if ser is None:
                ser = serial.Serial(port, 115200, timeout=0.5)
            raw = ser.readline()
        except (serial.SerialException, OSError) as e:   # usbip detached, cable pulled
            q.put((time.monotonic(), f"BTMIC HOST serial: {e}; retrying".encode()))
            if ser is not None:
                ser.close()
            ser = None
            time.sleep(1)
            continue
        if raw.startswith(b"A ") or raw.startswith(b"BTMIC"):
            q.put((time.monotonic(), raw))
    if ser is not None:
        ser.close()


class UsbStreamBoard:
    def __init__(self, port, stream_wav=None, log=print):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.pump_stop = ctx.Event()
        self.pump = ctx.Process(target=_pump, args=(port, self.q, self.pump_stop), daemon=True)
        self.pump.start()
        self.log = log
        self.lock = threading.Lock()
        self.chunks = collections.deque()   # (start_index, int16 array)
        self.end = 0                        # stream index just past the newest sample
        self.t_end = None                   # host time that newest chunk arrived
        self.arrivals = collections.deque() # (host time, stream index at its end), last minute
        self.expected = None                # next seq
        self.dropped = []                   # (start_index, samples) filled with zeros
        self.restarts = 0
        self.stat = None
        self.wav = sf.SoundFile(stream_wav, "w", RATE, 1, "PCM_16") if stream_wav else None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while self.t_end is None:
            if time.monotonic() - t0 > 15:
                self.close()
                raise SystemExit(f"no audio from the board on {port} in 15 s: is mic_stream.avm flashed, "
                                 "and the port attached (usbipd attach --wsl --auto-attach)?")
            time.sleep(0.1)

    # ------------------------------------------------------------ reading
    def _reader(self):
        while not self.stop.is_set():
            try:
                t, raw = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("A "):
                parts = line.split(" ", 2)
                if len(parts) == 3:
                    try:
                        self._add(int(parts[1]), np.frombuffer(base64.b64decode(parts[2]), "<i2"), t)
                    except (ValueError, base64.binascii.Error):
                        pass                         # a garbled line: counted as a gap by seq
            elif line.startswith("BTMIC"):
                if "STAT" in line:
                    self.stat = line
                else:
                    self.log(line)
                if "START" in line:
                    self.restarts += 1

    def _add(self, seq, pcm, t):
        with self.lock:
            if self.expected is not None and seq > self.expected:
                gap = (seq - self.expected) * len(pcm)  # lost lines: keep time true with zeros
                self._push(np.zeros(gap, np.int16))
                self.dropped.append((self.end - gap, gap))
            self.expected = seq + 1                  # (seq below expected: the board restarted)
            self._push(pcm)
            self.t_end = t                           # stamped on arrival by the pump process
            self.arrivals.append((self.t_end, self.end))
            while self.arrivals[0][0] < self.t_end - 60:
                self.arrivals.popleft()

    def _push(self, pcm):
        self.chunks.append((self.end, pcm))
        self.end += len(pcm)
        if self.wav:
            self.wav.write(pcm)
        while self.chunks and self.chunks[0][0] < self.end - KEEP_SECS * RATE:
            self.chunks.popleft()

    # ------------------------------------------------------------ cutting
    def now(self):
        """Stream index of 'now'. The stream's sample count is the board's clock (lost lines
        are filled in); a chunk can arrive late (the reader was busy, USB buffering) but never
        early, so host time = index / RATE + the smallest delay seen in the last minute."""
        with self.lock:
            offset = min(t - i / RATE for t, i in self.arrivals)
        return int((time.monotonic() - offset) * RATE)

    def span(self, a, b):
        """Samples [a, b) of the stream, and how many of them were lost (zeros)."""
        with self.lock:
            parts = [c for s, c in self.chunks if s + len(c) > a and s < b]
            first = next((s for s, c in self.chunks if s + len(c) > a), a)
            lost = sum(min(b, s + n) - max(a, s) for s, n in self.dropped if s < b and s + n > a)
        pcm = np.concatenate(parts) if parts else np.zeros(0, np.int16)
        return pcm[max(0, a - first): max(0, b - first)], lost

    def record(self, secs, on_start=None, pre=0.3):
        """-> int16 [n, 1]: from `pre` s before now to `secs` s after on_start()."""
        a = self.now() - int(pre * RATE)
        if on_start:
            on_start()
        b = a + int((pre + secs) * RATE)
        deadline = time.monotonic() + secs + 10
        while True:
            with self.lock:
                if self.end >= b:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("the board's stream stalled")
            time.sleep(0.05)
        pcm, self.last_lost = self.span(a, b)
        self.last_at = a                     # where in the stream (stream.wav) it starts
        return pcm.reshape(-1, 1)

    def close(self):
        self.stop.set()
        self.pump_stop.set()
        self.thread.join(timeout=2)
        self.pump.join(timeout=3)
        if self.pump.is_alive():
            self.pump.terminate()
        if self.wav:
            self.wav.close()


def align(rec, ref, before=0.25, after=0.6):
    """Where `ref` (the played audio) starts in `rec`, by cross-correlation.
    -> (start, end, score): the clip to keep, and how clear the match was (peak over the
    correlation's spread; above ~6 is a clear match, below ~4 a guess)."""
    x = rec.astype(np.float64)
    y = ref.astype(np.float64)
    if len(x) < len(y) or not y.any():
        return 0, len(rec), 0.0
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n)[: len(x) - len(y) + 1]
    k = int(np.argmax(np.abs(c)))
    score = float(np.abs(c[k]) / (np.std(c) + 1e-9))
    start = max(0, k - int(before * RATE))
    end = min(len(rec), k + len(ref) + int(after * RATE))
    return start, end, round(score, 1)
