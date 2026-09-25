"""Test vectors for the mmrt NIFs: an independent Python model of the arithmetic (the rounding
of mmrt/mmrt_ref.c), written out as the Erlang module src/mmrt_vectors.erl.

    python3 gen_vectors.py

Each case is {Name, Fun, Args, Expected}; an arg {packed, RowMajor, Rows, Cols} is replaced
on the device by mmrt:pack(RowMajor, Rows, Cols).
"""
import random
from pathlib import Path

rng = random.Random(1234)


def rnd(v, s):  # v * 2^-s, round half up (s > 0) / exact (s <= 0)
    return (v + (1 << (s - 1))) >> s if s > 0 else v << -s


def sat(v, lo=-128, hi=127):
    return max(lo, min(hi, v))


def ints(n, lo, hi):
    return [rng.randint(lo, hi) for _ in range(n)]


def b8(xs):
    return "<<" + ",".join(str(x) for x in xs) + ">>"


def b32(xs):
    return "<<" + ",".join(f"{x}:32/little-signed" for x in xs) + ">>"


def b4(xs):
    xs = xs + [0] * (len(xs) % 2)
    return "<<" + ",".join(str((xs[i] & 15) | ((xs[i + 1] & 15) << 4)) for i in range(0, len(xs), 2)) + ">>"


def pad16(n):
    return (n + 15) // 16 * 16


def pack_int8(w, R, C):  # [Rp/16][Cp][16]
    Rp, Cp = pad16(R), pad16(C)
    out = [0] * (Rp * Cp)
    for r in range(R):
        for c in range(C):
            out[(r // 16) * Cp * 16 + c * 16 + r % 16] = w[r * C + c]
    return out


def mat(w, x, R, C, shift=None):  # x: T rows of C
    T = len(x) // C
    y = []
    for t in range(T):
        for r in range(R):
            acc = sum(w[r * C + c] * x[t * C + c] for c in range(C))
            y.append(acc if shift is None else sat(rnd(acc, shift)))
    return y


def maxl1(w, R, C):
    return max(sum(abs(w[r * C + c]) for c in range(C)) for r in range(R))


cases = []


def case(name, fun, args, expected):
    cases.append(f'{{"{name}", {fun}, [{", ".join(args)}], {expected}}}')


# int8 matrix, odd shape (padding), small inputs: the SIMD path
R, C = 37, 45
w = ints(R * C, -128, 127)
x = ints(C, -20, 20)
case("pack int8 37x45", "pack", [f"{{int8, {b8(w)}}}", str(R), str(C)],
     f"{{int8_m16, {R}, {C}, {maxl1(w, R, C)}, {b8(pack_int8(w, R, C))}}}")
M = f"{{packed, {{int8, {b8(w)}}}, {R}, {C}}}"
for s in (0, 5, 9):
    case(f"matvec int8 shift {s}", "matvec", [M, f"{{int8, {b8(x)}}}", str(s)], f"{{int8, {b8(mat(w, x, R, C, s))}}}")
case("matvec int8 exact", "matvec", [M, f"{{int8, {b8(x)}}}"], f"{{int32, {b32(mat(w, x, R, C))}}}")
case("matvec int8 shift -2", "matvec", [M, f"{{int8, {b8(x)}}}", "-2"], f"{{int8, {b8(mat(w, x, R, C, -2))}}}")

# full-scale inputs: accumulators beyond the 20-bit lanes, so the exact C path
R, C = 20, 160
w = [rng.choice((-128, 127)) for _ in range(R * C)]
x = [rng.choice((-128, 127)) for _ in range(C)]
M = f"{{packed, {{int8, {b8(w)}}}, {R}, {C}}}"
case("matvec overflow-safe shift 10", "matvec", [M, f"{{int8, {b8(x)}}}", "10"], f"{{int8, {b8(mat(w, x, R, C, 10))}}}")
case("matvec overflow-safe exact", "matvec", [M, f"{{int8, {b8(x)}}}"], f"{{int32, {b32(mat(w, x, R, C))}}}")

# matmul, T = 5
R, C, T = 64, 128, 5
w = ints(R * C, -60, 60)
x = ints(T * C, -40, 40)
M = f"{{packed, {{int8, {b8(w)}}}, {R}, {C}}}"
case("matmul int8 T=5 shift 8", "matmul", [M, f"{{int8, {b8(x)}}}", "8"], f"{{int8, {b8(mat(w, x, R, C, 8))}}}")

# int4 weights (cb4_m16)
R, C, T = 40, 33, 3
w = ints(R * C, -8, 7)
x = ints(T * C, -128, 127)
M = f"{{packed, {{int4, {b4(w)}}}, {R}, {C}}}"
case("matmul int4 T=3 shift 4", "matmul", [M, f"{{int8, {b8(x)}}}", "4"], f"{{int8, {b8(mat(w, x, R, C, 4))}}}")
case("matvec int4 exact", "matvec", [M, f"{{int8, {b8(x[:C])}}}"], f"{{int32, {b32(mat(w, x[:C], R, C))}}}")

# elementwise and reductions
a, b = ints(50, -128, 127), ints(50, -128, 127)
case("dot int8", "dot", [f"{{int8, {b8(a)}}}", f"{{int8, {b8(b)}}}"], str(sum(p * q for p, q in zip(a, b))))
q = ints(50, -8, 7)
case("dot int8.int4", "dot", [f"{{int8, {b8(a)}}}", f"{{int4, {b4(q)}}}"], str(sum(p * r for p, r in zip(a, q))))
case("add int8", "add", [f"{{int8, {b8(a)}}}", f"{{int8, {b8(b)}}}"], f"{{int8, {b8([sat(p + r) for p, r in zip(a, b)])}}}")
big = ints(20, -2**31, 2**31 - 1)
big2 = ints(20, -2**31, 2**31 - 1)
case("add int32", "add", [f"{{int32, {b32(big)}}}", f"{{int32, {b32(big2)}}}"],
     f"{{int32, {b32([sat(p + r, -2**31, 2**31 - 1) for p, r in zip(big, big2)])}}}")
case("relu int8", "relu", [f"{{int8, {b8(a)}}}"], f"{{int8, {b8([max(0, p) for p in a])}}}")
case("relu int32", "relu", [f"{{int32, {b32(big)}}}"], f"{{int32, {b32([max(0, p) for p in big])}}}")
acc = ints(40, -300000, 300000)
for s in (12, 0, -3):
    case(f"requant shift {s}", "requant", [f"{{int32, {b32(acc)}}}", str(s)], f"{{int8, {b8([sat(rnd(v, s)) for v in acc])}}}")
case("argmax int8", "argmax", [f"{{int8, {b8(a)}}}"], str(a.index(max(a))))
case("argmax int32", "argmax", [f"{{int32, {b32(big)}}}"], str(big.index(max(big))))
ties = [3, 9, 1, 9, -4, 7, 9, 0]
top = sorted(range(len(ties)), key=lambda i: (-ties[i], i))[:4]
case("top_k ties", "top_k", [f"{{int8, {b8(ties)}}}", "4"], "[" + ",".join(f"{{{i},{ties[i]}}}" for i in top) + "]")
top = sorted(range(len(big)), key=lambda i: (-big[i], i))[:3]
case("top_k int32", "top_k", [f"{{int32, {b32(big)}}}", "3"], "[" + ",".join(f"{{{i},{big[i]}}}" for i in top) + "]")
odd = ints(7, -20, 20)
case("to_int4", "to_int4", [f"{{int8, {b8(odd)}}}"], f"{{int4, {b4([sat(v, -8, 7) for v in odd])}}}")
case("from_int4", "from_int4", [f"{{int4, {b4(q)}}}"], f"{{int8, {b8(q)}}}")

out = Path(__file__).parent / "src/mmrt_vectors.erl"
out.write_text("%% Generated by gen_vectors.py -- do not edit.\n-module(mmrt_vectors).\n-export([cases/0]).\n\n"
               "cases() -> [\n    " + ",\n    ".join(cases) + "\n].\n")
print(f"{len(cases)} cases -> {out} ({out.stat().st_size} bytes)")
