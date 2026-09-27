// MMRT NIFs for AtomVM: int8/int4 vector kernels over tagged binaries, module `mmrt`
// (Erlang) / `MMRT` (Elixir, delegating). The Erlang docs are in atomvm/lib/mmrt/src/mmrt.erl.
//
// Data:
//   {int8, Bin}    one int8 per byte
//   {int4, Bin}    signed nibbles, low nibble first (2 per byte)
//   {int32, Bin}   little-endian int32 (accumulators)
//   {int8_m16, Rows, Cols, MaxRowL1, Bin}   matrices packed by pack/3 for the PIE kernel:
//   {cb4_m16,  Rows, Cols, MaxRowL1, Bin}   MMRT's 1x1-conv weight layouts (mmrt/mmrt.h), padded
//                                           to multiples of 16; MaxRowL1 = max_r sum_c |W[r][c]|
//
// matvec/matmul run on a vector unit when there is one: on the ESP32-S3's PIE when the
// accumulators provably fit its 20-bit lanes (MaxRowL1 * max|x| < 2^19), on the ESP32-P4's as
// row dot products into a 40-bit accumulator (always), else on an exact C path: results are
// the same either way.
// NIF arguments are not GC roots unless passed as such: every NIF reads its inputs
// before it allocates result terms (top_k re-fetches after a rooted GC).
// Everything runs on the calling scheduler (no dirty NIFs in AtomVM), so each call is
// capped at MMRT_SYNC_MAX_MACS multiply-adds; bigger work raises `too_big`.
#include <sdkconfig.h>
#ifdef CONFIG_AVM_ENABLE_MMRT_NIFS

#include <stdlib.h>
#include <string.h>

#include <atom.h>
#include <defaultatoms.h>
#include <globalcontext.h>
#include <interop.h>
#include <memory.h>
#include <nifs.h>
#include <portnifloader.h>
#include <term.h>
#include <utils.h>

#include <esp_heap_caps.h>

#include "mmrt.h"
#ifdef MMRT_S3
#include "mmrt_s3.h"

// in mmrt_s3.S: one CB4 group's index rows -> int8 (dst, idx 4-byte aligned)
void mmrt_s3_cb4_rows(int8_t *dst, const uint8_t *idx, const uint32_t *tab32, int C);
#endif
#ifdef MMRT_P4
#include "mmrt_port.h"  // mmrt_dot_s8: the P4's vector dot product (mmrt_p4.S)
#endif

#define MMRT_SYNC_MAX_MACS (4 * 1024 * 1024)  // ~2 ms on PIE, ~30 ms on the C path
#define MAX_DIM 65535
#define QACC_LIMIT (1 << 19)

static const char *const A_INT8 = "\x04" "int8";
static const char *const A_INT4 = "\x04" "int4";
static const char *const A_INT32 = "\x05" "int32";
static const char *const A_INT8_M16 = "\x08" "int8_m16";
static const char *const A_CB4_M16 = "\x07" "cb4_m16";
static const char *const A_TOO_BIG = "\x07" "too_big";

enum { V_INT8, V_INT4, V_INT32 };
enum { M_INT8, M_CB4 };

typedef struct {
    int type;
    const uint8_t *p;
    size_t bytes, n;  // n: elements
} vec_t;

typedef struct {
    int fmt;
    int rows, cols, npad, cpad;
    int32_t maxl1;
    const uint8_t *p;
    size_t gbytes;  // bytes per 16-row group
} mat_t;

static inline int pad16(int n) { return (n + 15) & ~15; }
static inline int8_t sat8(int64_t v) { return v > 127 ? 127 : (v < -128 ? -128 : (int8_t) v); }
static inline int32_t sat32(int64_t v) { return v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : (int32_t) v); }
// v * 2^-s rounded half up (s > 0), or v * 2^-s exactly (s <= 0): mmrt_ref.c's rounding.
// Callers keep |v| < 2^31 and -31 <= s <= 62.
static inline int64_t rshift_rnd(int64_t v, int s)
{
    return s > 0 ? (v + ((int64_t) 1 << (s - 1))) >> s : v * ((int64_t) 1 << -s);
}
static inline int nib(const uint8_t *p, size_t i) { int v = (p[i >> 1] >> ((i & 1) * 4)) & 15; return v < 8 ? v : v - 16; }
static inline int32_t rd32(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
static inline void wr32(uint8_t *p, int32_t v) { memcpy(p, &v, 4); }

static term atom(Context *ctx, const char *a) { return globalcontext_make_atom(ctx->global, a); }

static bool get_int(term t, avm_int_t lo, avm_int_t hi, avm_int_t *out)
{
    if (!term_is_integer(t)) return false;
    avm_int_t v = term_to_int(t);
    if (v < lo || v > hi) return false;
    *out = v;
    return true;
}

// {int8 | int4 | int32, Bin}. Pointers into heap binaries go stale on GC: fetch after allocating.
static bool get_vec(Context *ctx, term t, vec_t *v)
{
    if (!term_is_tuple(t) || term_get_tuple_arity(t) != 2) return false;
    term tag = term_get_tuple_element(t, 0), bin = term_get_tuple_element(t, 1);
    if (!term_is_binary(bin)) return false;
    v->p = (const uint8_t *) term_binary_data(bin);
    v->bytes = term_binary_size(bin);
    if (tag == atom(ctx, A_INT8)) {
        v->type = V_INT8;
        v->n = v->bytes;
    } else if (tag == atom(ctx, A_INT4)) {
        v->type = V_INT4;
        v->n = v->bytes * 2;
    } else if (tag == atom(ctx, A_INT32)) {
        if (v->bytes % 4) return false;
        v->type = V_INT32;
        v->n = v->bytes / 4;
    } else {
        return false;
    }
    return true;
}

static int32_t vec_at(const vec_t *v, size_t i)
{
    switch (v->type) {
        case V_INT8: return (int8_t) v->p[i];
        case V_INT4: return nib(v->p, i);
        default: return rd32(v->p + 4 * i);
    }
}

static bool get_mat(Context *ctx, term t, mat_t *m)
{
    if (!term_is_tuple(t) || term_get_tuple_arity(t) != 5) return false;
    term tag = term_get_tuple_element(t, 0), bin = term_get_tuple_element(t, 4);
    avm_int_t r, c, l1;
    if (!get_int(term_get_tuple_element(t, 1), 1, MAX_DIM, &r) || !get_int(term_get_tuple_element(t, 2), 1, MAX_DIM, &c)
        || !term_is_any_integer(term_get_tuple_element(t, 3)) || !term_is_binary(bin)) {
        return false;
    }
    int64_t l1_64 = term_maybe_unbox_int64(term_get_tuple_element(t, 3));
    if (l1_64 < 0 || l1_64 > INT32_MAX) return false;
    l1 = (avm_int_t) l1_64;
    m->rows = r;
    m->cols = c;
    m->npad = pad16(r);
    m->cpad = pad16(c);
    m->maxl1 = l1;
    if (tag == atom(ctx, A_INT8_M16)) {
        m->fmt = M_INT8;
    } else if (tag == atom(ctx, A_CB4_M16)) {
        m->fmt = M_CB4;
    } else {
        return false;
    }
    m->gbytes = mmrt_wgroup_bytes(m->cpad, m->fmt == M_CB4 ? MMRT_W_CB4 : MMRT_W_INT8);
    if (term_binary_size(bin) != (size_t) (m->npad / 16) * m->gbytes) return false;
    m->p = (const uint8_t *) term_binary_data(bin);
    return true;
}

// 16-aligned scratch: internal RAM when it is small enough and available, else PSRAM.
// (on the P4, DMA-capable internal RAM: that leaves out its RTC RAM, which the vector unit can't read)
#ifdef MMRT_P4
#define FAST_CAPS (MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA)
#else
#define FAST_CAPS (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#endif
static void *scratch(size_t bytes, bool prefer_internal)
{
    void *p = NULL;
    if (prefer_internal && bytes <= 32 * 1024) p = heap_caps_aligned_alloc(16, bytes, FAST_CAPS);
    if (!p) p = heap_caps_aligned_alloc(16, bytes ? bytes : 16, MALLOC_CAP_DEFAULT);
    return p;
}

// One 16-row group of a packed matrix as [cpad][16] int8 in dst (16-aligned). tab32: 1 KB
// scratch for the CB4 decoder (the scheduler stacks are 3.5 KB).
static void load_group(const mat_t *m, int g, int8_t *dst, uint32_t *tab32)
{
    const uint8_t *src = m->p + (size_t) g * m->gbytes;
    if (m->fmt == M_INT8) {
        memcpy(dst, src, m->gbytes);
        return;
    }
    const uint8_t *idx = src + 256;
#ifdef MMRT_S3
    if (((uintptr_t) idx & 3) == 0) {
        for (int i = 0; i < 256; i++) tab32[i] = (uint32_t) src[i] << (8 * ((i >> 4) & 3));
        mmrt_s3_cb4_rows(dst, idx, tab32, m->cpad);
        return;
    }
#else
    (void) tab32;
#endif
    for (int c = 0; c < m->cpad; c++, idx += 8, dst += 16) {  // portable decode (any alignment)
        for (int j = 0; j < 8; j++) {
            dst[2 * j] = (int8_t) src[(2 * j) * 16 + (idx[j] & 15)];
            dst[2 * j + 1] = (int8_t) src[(2 * j + 1) * 16 + (idx[j] >> 4)];
        }
    }
}

// ------------------------------------------------------------------------------ results

static term make_vec(Context *ctx, const char *tag, const void *data, size_t bytes)
{
    if (UNLIKELY(memory_ensure_free(ctx, term_binary_heap_size(bytes) + TUPLE_SIZE(2)) != MEMORY_GC_OK)) {
        return term_invalid_term();
    }
    term bin = term_create_uninitialized_binary(bytes, &ctx->heap, ctx->global);
    if (term_is_invalid_term(bin)) return bin;
    memcpy((void *) term_binary_data(bin), data, bytes);
    term t = term_alloc_tuple(2, &ctx->heap);
    term_put_tuple_element(t, 0, atom(ctx, tag));
    term_put_tuple_element(t, 1, bin);
    return t;
}

#define RETURN_VEC(tag, data, bytes, tofree)                    \
    do {                                                         \
        term r_ = make_vec(ctx, (tag), (data), (bytes));         \
        free(tofree);                                            \
        if (term_is_invalid_term(r_)) RAISE_ERROR(OUT_OF_MEMORY_ATOM); \
        return r_;                                               \
    } while (0)

static term make_int(Context *ctx, int64_t v)
{
    if (UNLIKELY(memory_ensure_free(ctx, BOXED_INT64_SIZE) != MEMORY_GC_OK)) return term_invalid_term();
    return term_make_maybe_boxed_int64(v, &ctx->heap);
}

// ------------------------------------------------------------------------------ NIFs

// pack({int8, RowMajor} | {int4, RowMajor}, Rows, Cols) -> {int8_m16 | cb4_m16, Rows, Cols, MaxRowL1, Bin}
static term nif_pack(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t v;
    avm_int_t R, C;
    if (!get_vec(ctx, argv[0], &v) || v.type == V_INT32 || !get_int(argv[1], 1, MAX_DIM, &R) || !get_int(argv[2], 1, MAX_DIM, &C)) {
        RAISE_ERROR(BADARG_ATOM);
    }
    size_t n = (size_t) R * C;
    if (v.type == V_INT8 ? v.n != n : (v.bytes != (n + 1) / 2)) RAISE_ERROR(BADARG_ATOM);
    const int np = pad16(R), cp = pad16(C);
    const bool cb4 = v.type == V_INT4;
    const size_t gb = mmrt_wgroup_bytes(cp, cb4 ? MMRT_W_CB4 : MMRT_W_INT8), total = (size_t) (np / 16) * gb;

    uint8_t *out = calloc(1, total);
    if (!out) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    int32_t maxl1 = 0;
    for (int r = 0; r < R; r++) {
        int32_t l1 = 0;
        uint8_t *g = out + (size_t) (r / 16) * gb;
        const int lane = r % 16;
        for (int c = 0; c < C; c++) {
            size_t i = (size_t) r * C + c;
            int w = cb4 ? nib(v.p, i) : (int8_t) v.p[i];
            l1 += w < 0 ? -w : w;
            if (cb4) g[256 + (size_t) c * 8 + lane / 2] |= (uint8_t) ((w & 15) << ((lane & 1) * 4));
            else g[(size_t) c * 16 + lane] = (uint8_t) w;
        }
        if (l1 > maxl1) maxl1 = l1;
    }
    if (cb4) {  // identity codebook: level k is the signed nibble k
        for (int g = 0; g < np / 16; g++)
            for (int i = 0; i < 256; i++) out[(size_t) g * gb + i] = (uint8_t) (int8_t) ((i & 15) < 8 ? (i & 15) : (i & 15) - 16);
    }
    if (UNLIKELY(memory_ensure_free(ctx, term_binary_heap_size(total) + TUPLE_SIZE(5)) != MEMORY_GC_OK)) {
        free(out);
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    term bin = term_create_uninitialized_binary(total, &ctx->heap, ctx->global);
    if (term_is_invalid_term(bin)) {
        free(out);
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    memcpy((void *) term_binary_data(bin), out, total);
    free(out);
    term t = term_alloc_tuple(5, &ctx->heap);
    term_put_tuple_element(t, 0, atom(ctx, cb4 ? A_CB4_M16 : A_INT8_M16));
    term_put_tuple_element(t, 1, term_from_int(R));
    term_put_tuple_element(t, 2, term_from_int(C));
    term_put_tuple_element(t, 3, term_from_int(maxl1));  // <= 65535 * 128, a small int
    term_put_tuple_element(t, 4, bin);
    return t;
}

// matmul(M, {int8, X}, Shift) -> {int8, Y}: X is T rows of Cols, Y is T rows of Rows,
// Y = sat8(round_half_up((X . W^T) >> Shift)).
// matvec(M, {int8, X}) -> {int32, Y}: one row, exact accumulators.
static term do_matmul(Context *ctx, term mt, term xt, bool to_int32, avm_int_t shift)
{
    mat_t m;
    vec_t x;
    if (!get_mat(ctx, mt, &m) || !get_vec(ctx, xt, &x) || x.type != V_INT8 || x.n % m.cols) RAISE_ERROR(BADARG_ATOM);
    const int T = (int) (x.n / m.cols);
    if (to_int32 && T != 1) RAISE_ERROR(BADARG_ATOM);
    if ((uint64_t) T * m.npad * m.cpad > MMRT_SYNC_MAX_MACS) RAISE_ERROR(atom(ctx, A_TOO_BIG));

    int maxabs = 0;
    for (size_t i = 0; i < x.n; i++) {
        int a = (int8_t) x.p[i];
        if (a < 0) a = -a;
        if (a > maxabs) maxabs = a;
    }
#if defined(MMRT_S3)
    const bool pie = !to_int32 && shift >= 0 && shift <= 20 && (int64_t) m.maxl1 * maxabs < QACC_LIMIT;
#elif defined(MMRT_P4)  // its kernel's rounding covers shifts up to 13
    const bool pie = !to_int32 && shift >= 0 && shift <= 13 && (int64_t) m.maxl1 * maxabs < QACC_LIMIT
                     && mmrt_p4_matmul_ok();
#else
    const bool pie = false;
    (void) maxabs;
#endif
#ifdef MMRT_P4
    // the P4: each group's 16 rows repacked contiguous, each output one dot product (exact, any size)
    static int dot_ok = -1;
    if (dot_ok < 0) dot_ok = mmrt_dot_p4_check();
    int8_t *rows = dot_ok && !pie ? scratch((size_t) m.cpad * 16, true) : NULL;  // (int32 results, big accumulators)
#endif

    const size_t xb = (size_t) T * m.cpad, wb = (size_t) m.cpad * 16;
    const size_t yb = to_int32 ? (size_t) m.rows * 4 : (size_t) T * (pie ? m.npad : m.rows);
    int8_t *xs = scratch(xb, true), *ws = scratch(wb, true), *ys = scratch(yb, false);
    uint32_t *tab = m.fmt == M_CB4 ? malloc(1024) : NULL;
    if (!xs || !ws || !ys || (m.fmt == M_CB4 && !tab)) {
        heap_caps_free(xs);
        heap_caps_free(ws);
        heap_caps_free(ys);
        free(tab);
#ifdef MMRT_P4
        heap_caps_free(rows);
#endif
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    memset(xs, 0, xb);  // padded columns multiply as zero
    for (int t = 0; t < T; t++) memcpy(xs + (size_t) t * m.cpad, x.p + (size_t) t * m.cols, m.cols);

    for (int g = 0; g < m.npad / 16; g++) {
        load_group(&m, g, ws, tab);
        if (pie) {
#if defined(MMRT_S3)
            mmrt_s3_matmul_group(xs, T, m.cpad, ws, (int) shift, 0, ys + g * 16, m.npad);
#elif defined(MMRT_P4)
            mmrt_p4_matmul_group(xs, T, m.cpad, ws, (int) shift, 0, ys + g * 16, m.npad);
#endif
            continue;
        }
        const int lanes = m.rows - g * 16 < 16 ? m.rows - g * 16 : 16;
#ifdef MMRT_P4
        if (rows)
            for (int c = 0; c < m.cpad; c++)
                for (int i = 0; i < 16; i++) rows[(size_t) i * m.cpad + c] = ws[c * 16 + i];
#endif
        for (int t = 0; t < T; t++) {
            const int8_t *xr = xs + (size_t) t * m.cpad;
            int32_t acc[16] = {0};  // |acc| <= 65535 * 128 * 128 < 2^31
#ifdef MMRT_P4
            if (rows) {
                for (int i = 0; i < lanes; i++) acc[i] = mmrt_dot_s8(xr, rows + (size_t) i * m.cpad, m.cpad / 16);
            } else
#endif
            for (int c = 0; c < m.cols; c++) {
                const int32_t xv = xr[c];
                const int8_t *wc = ws + c * 16;
                for (int i = 0; i < 16; i++) acc[i] += xv * wc[i];
            }
            for (int i = 0; i < lanes; i++) {
                const int r = g * 16 + i;
                if (to_int32) wr32((uint8_t *) ys + 4 * r, acc[i]);
                else ys[(size_t) t * m.rows + r] = sat8(rshift_rnd(acc[i], (int) shift));
            }
        }
    }
    if (pie && m.npad != m.rows) {  // drop the padded outputs
        for (int t = 1; t < T; t++) memmove(ys + (size_t) t * m.rows, ys + (size_t) t * m.npad, m.rows);
    }
    heap_caps_free(xs);
    heap_caps_free(ws);
    free(tab);
#ifdef MMRT_P4
    heap_caps_free(rows);
#endif
    term r = make_vec(ctx, to_int32 ? A_INT32 : A_INT8, ys, to_int32 ? yb : (size_t) T * m.rows);
    heap_caps_free(ys);
    if (term_is_invalid_term(r)) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    return r;
}

static term nif_matvec2(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    return do_matmul(ctx, argv[0], argv[1], true, 0);
}

static term nif_matmul3(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    avm_int_t shift;
    if (!get_int(argv[2], -31, 62, &shift)) RAISE_ERROR(BADARG_ATOM);
    return do_matmul(ctx, argv[0], argv[1], false, shift);
}

// dot(A, B) -> Integer, any two int8/int4/int32 vectors of the same length
static term nif_dot(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a, b;
    if (!get_vec(ctx, argv[0], &a) || !get_vec(ctx, argv[1], &b) || a.n != b.n) RAISE_ERROR(BADARG_ATOM);
    if (a.n > MMRT_SYNC_MAX_MACS) RAISE_ERROR(atom(ctx, A_TOO_BIG));
    int64_t s = 0;
    if (a.type == V_INT8 && b.type == V_INT8) {
        const int8_t *p = (const int8_t *) a.p, *q = (const int8_t *) b.p;
        int32_t acc = 0;  // flushed before it could overflow: 2^16 * 2^14 = 2^30
        for (size_t i = 0; i < a.n; i++) {
            acc += p[i] * q[i];
            if ((i & 0xffff) == 0xffff) {
                s += acc;
                acc = 0;
            }
        }
        s += acc;
    } else {
        for (size_t i = 0; i < a.n; i++) s += (int64_t) vec_at(&a, i) * vec_at(&b, i);
    }
    term r = make_int(ctx, s);
    if (term_is_invalid_term(r)) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    return r;
}

// add(A, B) -> saturating elementwise sum, both int8 or both int32
static term nif_add(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a, b;
    if (!get_vec(ctx, argv[0], &a) || !get_vec(ctx, argv[1], &b) || a.type != b.type || a.type == V_INT4 || a.n != b.n) {
        RAISE_ERROR(BADARG_ATOM);
    }
    uint8_t *y = malloc(a.bytes ? a.bytes : 1);
    if (!y) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    for (size_t i = 0; i < a.n; i++) {
        if (a.type == V_INT8) y[i] = (uint8_t) sat8((int8_t) a.p[i] + (int8_t) b.p[i]);
        else wr32(y + 4 * i, sat32((int64_t) rd32(a.p + 4 * i) + rd32(b.p + 4 * i)));
    }
    RETURN_VEC(a.type == V_INT8 ? A_INT8 : A_INT32, y, a.bytes, y);
}

// relu(V) -> max(V, 0), int8 or int32
static term nif_relu(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a;
    if (!get_vec(ctx, argv[0], &a) || a.type == V_INT4) RAISE_ERROR(BADARG_ATOM);
    uint8_t *y = malloc(a.bytes ? a.bytes : 1);
    if (!y) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    for (size_t i = 0; i < a.n; i++) {
        if (a.type == V_INT8) y[i] = (int8_t) a.p[i] < 0 ? 0 : a.p[i];
        else wr32(y + 4 * i, rd32(a.p + 4 * i) < 0 ? 0 : rd32(a.p + 4 * i));
    }
    RETURN_VEC(a.type == V_INT8 ? A_INT8 : A_INT32, y, a.bytes, y);
}

// requant({int32, B}, Shift) -> {int8, sat8(round_half_up(B >> Shift))} (Shift < 0 multiplies)
static term nif_requant(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a;
    avm_int_t shift;
    if (!get_vec(ctx, argv[0], &a) || a.type != V_INT32 || !get_int(argv[1], -31, 62, &shift)) RAISE_ERROR(BADARG_ATOM);
    uint8_t *y = malloc(a.n ? a.n : 1);
    if (!y) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    for (size_t i = 0; i < a.n; i++) y[i] = (uint8_t) sat8(rshift_rnd(rd32(a.p + 4 * i), (int) shift));
    RETURN_VEC(A_INT8, y, a.n, y);
}

// argmax(V) -> 0-based index of the first largest element
static term nif_argmax(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a;
    if (!get_vec(ctx, argv[0], &a) || a.n == 0) RAISE_ERROR(BADARG_ATOM);
    size_t best = 0;
    int32_t bv = vec_at(&a, 0);
    for (size_t i = 1; i < a.n; i++) {
        int32_t v = vec_at(&a, i);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return term_from_int((avm_int_t) best);
}

// top_k(V, K) -> [{Index, Value}], largest first (ties: lower index first)
static term nif_top_k(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a;
    avm_int_t K;
    if (!get_vec(ctx, argv[0], &a) || !get_int(argv[1], 0, 4096, &K)) RAISE_ERROR(BADARG_ATOM);
    if ((size_t) K > a.n) K = (avm_int_t) a.n;
    uint32_t *idx = malloc(sizeof(uint32_t) * (K ? K : 1));
    if (!idx) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    int n = 0;  // insertion into a sorted top list: O(N*K), fine for the K this is for
    for (size_t i = 0; i < a.n; i++) {
        int32_t v = vec_at(&a, i);
        if (n == K && (K == 0 || v <= vec_at(&a, idx[n - 1]))) continue;
        int j = n < K ? n++ : n - 1;
        while (j > 0 && vec_at(&a, idx[j - 1]) < v) {
            idx[j] = idx[j - 1];
            j--;
        }
        idx[j] = (uint32_t) i;
    }
    // argv[0] is a root: the GC may move a heap binary, so re-fetch the data after it
    size_t words = (size_t) n * (CONS_SIZE + TUPLE_SIZE(2) + BOXED_INT64_SIZE);
    if (UNLIKELY(memory_ensure_free_with_roots(ctx, words, 1, argv, MEMORY_CAN_SHRINK) != MEMORY_GC_OK)) {
        free(idx);
        RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    }
    get_vec(ctx, argv[0], &a);
    term list = term_nil();
    for (int j = n - 1; j >= 0; j--) {
        term t = term_alloc_tuple(2, &ctx->heap);
        term_put_tuple_element(t, 0, term_from_int((avm_int_t) idx[j]));
        term_put_tuple_element(t, 1, term_make_maybe_boxed_int64(vec_at(&a, idx[j]), &ctx->heap));
        list = term_list_prepend(t, list, &ctx->heap);
    }
    free(idx);
    return list;
}

// to_int4({int8, B}) -> {int4, clamp(B, -8, 7) packed}; from_int4({int4, B}) -> {int8, B unpacked}
static term nif_to_int4(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a;
    if (!get_vec(ctx, argv[0], &a) || a.type != V_INT8) RAISE_ERROR(BADARG_ATOM);
    size_t nb = (a.n + 1) / 2;
    uint8_t *y = calloc(1, nb ? nb : 1);
    if (!y) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    for (size_t i = 0; i < a.n; i++) {
        int v = (int8_t) a.p[i];
        v = v < -8 ? -8 : (v > 7 ? 7 : v);
        y[i >> 1] |= (uint8_t) ((v & 15) << ((i & 1) * 4));
    }
    RETURN_VEC(A_INT4, y, nb, y);
}

static term nif_from_int4(Context *ctx, int argc, term argv[])
{
    UNUSED(argc);
    vec_t a;
    if (!get_vec(ctx, argv[0], &a) || a.type != V_INT4) RAISE_ERROR(BADARG_ATOM);
    uint8_t *y = malloc(a.n ? a.n : 1);
    if (!y) RAISE_ERROR(OUT_OF_MEMORY_ATOM);
    for (size_t i = 0; i < a.n; i++) y[i] = (uint8_t) nib(a.p, i);
    RETURN_VEC(A_INT8, y, a.n, y);
}

#define NIF(name, fn) static const struct Nif name##_nif = {.base.type = NIFFunctionType, .nif_ptr = fn}
NIF(pack, nif_pack);
NIF(matvec2, nif_matvec2);
NIF(matmul3, nif_matmul3);
NIF(dot, nif_dot);
NIF(add, nif_add);
NIF(relu, nif_relu);
NIF(requant, nif_requant);
NIF(argmax, nif_argmax);
NIF(top_k, nif_top_k);
NIF(to_int4, nif_to_int4);
NIF(from_int4, nif_from_int4);

static const struct {
    const char *name;  // "fun/arity"
    const struct Nif *nif;
} NIFS[] = {
    {"pack/3", &pack_nif}, {"matvec/2", &matvec2_nif}, {"matvec/3", &matmul3_nif}, {"matmul/3", &matmul3_nif},
    {"dot/2", &dot_nif}, {"add/2", &add_nif}, {"relu/1", &relu_nif}, {"requant/2", &requant_nif},
    {"argmax/1", &argmax_nif}, {"top_k/2", &top_k_nif}, {"to_int4/1", &to_int4_nif}, {"from_int4/1", &from_int4_nif},
};

// The Elixir module MMRT delegates to :mmrt, so only "mmrt:..." names resolve here.
static const struct Nif *mmrt_get_nif(const char *name)
{
    if (strncmp(name, "mmrt:", 5)) return NULL;
    for (size_t i = 0; i < sizeof(NIFS) / sizeof(NIFS[0]); i++) {
        if (!strcmp(name + 5, NIFS[i].name)) return NIFS[i].nif;
    }
    return NULL;
}

REGISTER_NIF_COLLECTION(mmrt, NULL, NULL, mmrt_get_nif)

#endif
