#!/usr/bin/env python3
"""Requantize a sanoTTS nano-runtime export (float32 or int8) for the device.

  tools/sanotts_voice.py <header> <front blob> <decoder blob> <out dir> --fmt int8|q4

int8: sanoTTS's own device format (NANO_WEIGHT_FORMAT 0), per-row symmetric scales.
q4:   BabyTalk's 4-bit format (NANO_WEIGHT_FORMAT 2; the row layout is in
      components/sanotts/snt_q4.h): weights -7..7 with a float32 scale per 16 inputs, each
      scale chosen to minimise its group's squared rounding error. The row's own scale (the
      SCALE region) is 1. Everything else is copied as is.
Writes front.bin, model.bin, nano_q8_meta.h (the header with the new offsets) and voice.bin,
the two blobs as BabyTalk's "voice" flash partition holds them (components/sanotts/tts_engine.c):
four little-endian uint32 (magic "SNTV", NANO_WEIGHT_FORMAT, front bytes, decoder bytes), the
front at 64, the decoder after it at a multiple of 16."""
import argparse, pathlib, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('header'); ap.add_argument('front'); ap.add_argument('dec'); ap.add_argument('out')
ap.add_argument('--fmt', choices=('int8', 'q4'), required=True)
a = ap.parse_args()
hdr = pathlib.Path(a.header).read_text()
defs = {k: int(v) for k, v in re.findall(r'#define (\w+) (-?\d+)', hdr)}
SRC = defs.get('NANO_WEIGHT_FORMAT', 0)
assert SRC in (0, 1), 'expects a float32 or int8 export'
ES = 4 if SRC == 1 else 1

def q_int8(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8).tobytes(), s.astype(np.float32).reshape(-1)

def q_q4(w):
    rows, n = w.shape
    g = w.reshape(rows, n // 16, 16).astype(np.float64)
    s0 = np.abs(g).max(axis=2, keepdims=True) / 7.0
    s0[s0 == 0] = 1.0
    best, bs = None, s0
    for f in np.linspace(0.5, 1.0, 26):
        s = s0 * f
        e = ((np.clip(np.round(g / s), -7, 7) * s - g) ** 2).sum(axis=2, keepdims=True)
        if best is None: best, bs = e, s.copy()
        else:
            m = e < best; best = np.where(m, e, best); bs = np.where(m, s, bs)
    bs = bs.astype(np.float32)
    u = (np.clip(np.round(g / bs.astype(np.float64)), -7, 7) + 8).astype(np.uint8).reshape(rows, n)
    assert n % 16 == 0 and n >= 32
    blocks = n // 32
    out = np.zeros((rows, n), np.uint8)
    for b in range(blocks):
        out[:, 16 * b:16 * b + 16] = u[:, 32 * b:32 * b + 16] | (u[:, 32 * b + 16:32 * b + 32] << 4)
    if n % 32:
        out[:, 16 * blocks:16 * blocks + 16] = u[:, 32 * blocks:]
    sat = 16 * ((n + 31) // 32)
    out[:, sat:sat + n // 4] = bs.reshape(rows, n // 16).view(np.uint8).reshape(rows, n // 4)
    return out.tobytes(), np.ones(rows, np.float32)

def convert(prefix, blob):
    offs = sorted((v, k) for k, v in defs.items() if k.startswith(prefix + '_'))
    new, out, pos, scale_for = {}, bytearray(), 0, {}
    for i, (off, k) in enumerate(offs):
        end = offs[i + 1][0] if i + 1 < len(offs) else len(blob)
        chunk = blob[off:end]
        m = re.fullmatch(prefix + r'_(\w+)_W8', k)
        if m:
            n16 = defs['NANO_%s_N16' % m.group(1)]
            so = defs['%s_%s_SCALE' % (prefix, m.group(1))]
            rows = (so - off) // (ES * n16)
            if SRC == 1:
                w = np.frombuffer(chunk[:rows * n16 * 4], np.float32).reshape(rows, n16)
            else:   # int8 rows times their scales
                w = np.frombuffer(chunk[:rows * n16], np.int8).reshape(rows, n16).astype(np.float32) \
                    * np.frombuffer(blob[so:so + 4 * rows], np.float32)[:, None]
            data, sc = (q_int8 if a.fmt == 'int8' else q_q4)(w)
            scale_for[m.group(1)] = sc
            chunk = data + b'\0' * ((-len(data)) % 16)
        m = re.fullmatch(prefix + r'_(\w+)_SCALE', k)
        if m:
            sc = scale_for[m.group(1)]
            assert len(chunk) >= sc.nbytes and (SRC == 0 or np.all(np.frombuffer(chunk[:sc.nbytes], np.float32) == 1.0))
            chunk = sc.tobytes() + chunk[sc.nbytes:]
        assert pos % 16 == 0
        new[k] = pos
        out += chunk
        pos += len(chunk)
    return new, bytes(out)

o = pathlib.Path(a.out); o.mkdir(parents=True, exist_ok=True)
nf, front = convert('NOFF', pathlib.Path(a.front).read_bytes())
nd, dec = convert('DOFF', pathlib.Path(a.dec).read_bytes())
(o / 'front.bin').write_bytes(front); (o / 'model.bin').write_bytes(dec)
new = {**nf, **nd, 'NANO_FRONT_BYTES': len(front), 'NANO_DEC_BYTES': len(dec), 'NANO_WEIGHT_FORMAT': 0 if a.fmt == 'int8' else 2}
text = re.sub(r'#define (\w+) (-?\d+)', lambda m: '#define %s %d' % (m.group(1), new.get(m.group(1), int(m.group(2)))), hdr)
text = text.replace('front_f32.bin', 'front.bin').replace('model_f32.bin', 'model.bin')
if '#define NANO_WEIGHT_FORMAT' not in text:
    text = text.replace('#define NANO_FORMAT_VERSION', '#define NANO_WEIGHT_FORMAT %d\n#define NANO_FORMAT_VERSION' % new['NANO_WEIGHT_FORMAT'], 1)
(o / 'nano_q8_meta.h').write_text(text)
hdr_v = np.array([0x56544E53, new['NANO_WEIGHT_FORMAT'], len(front), len(dec)], '<u4').tobytes()
body = hdr_v.ljust(64, b'\0') + front
(o / 'voice.bin').write_bytes(body + b'\0' * ((-len(body)) % 16) + dec)
print('%s: front %d B, decoder %d B' % (a.fmt, len(front), len(dec)))
