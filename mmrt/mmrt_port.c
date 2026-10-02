#include "mmrt_port.h"

#include <stddef.h>
#include <string.h>

// the reference's output rounding: v * 2^-shift rounded half up, ReLU, saturated to int8
static inline int8_t out8(int32_t acc, int shift, int relu)
{
    int64_t v = shift > 0 ? ((int64_t)acc + ((int64_t)1 << (shift - 1))) >> shift : (int64_t)acc * ((int64_t)1 << -shift);
    if (relu && v < 0) v = 0;
    return v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v);
}

void mmrt_dwconv_port(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                      int shift, int relu, int8_t *y, int T_out)
{
    for (int g = 0; g < C / 16; g++) {
        const int8_t *wg = w + (size_t)g * K * 16;
        for (int t = 0; t < T_out; t++) {
            int32_t acc[16] = {0};
            for (int k = 0; k < K; k++) {
                const int ti = t * stride + k - pad;
                if (ti < 0 || ti >= T_in) continue;
                const int8_t *xr = x + (size_t)ti * C + g * 16, *wk = wg + k * 16;
                for (int j = 0; j < 16; j++) acc[j] += xr[j] * wk[j];
            }
            int8_t *yt = y + (size_t)t * C + g * 16;
            for (int j = 0; j < 16; j++) yt[j] = out8(acc[j], shift, relu);
        }
    }
}

void mmrt_conv1x1_port(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                       int stride, int shift, int relu, int8_t *y, int T_out)
{
    (void)T_in;
    for (int g = 0; g < N / 16; g++) {
        const int8_t *wg = w + (size_t)g * C * 16;     // this group's 16 x C weights: cached across frames
        for (int t = 0; t < T_out; t++) {
            const int8_t *xt = x + (size_t)t * stride * C;
            int32_t acc[16];
            for (int j = 0; j < 16; j++) acc[j] = bias ? bias[g * 16 + j] : 0;
            const int8_t *wc = wg;
            for (int c = 0; c < C; c++, wc += 16) {
                const int32_t xc = xt[c];
                if (!xc) continue;
                for (int j = 0; j < 16; j++) acc[j] += xc * wc[j];
            }
            int8_t *yt = y + (size_t)t * N + g * 16;
            for (int j = 0; j < 16; j++) yt[j] = out8(acc[j], shift, relu);
        }
    }
}

#ifdef MMRT_P4
static int cb4_p4_ok(void);
void mmrt_p4_cb4_rows(int8_t *dst, const uint8_t *idx, const int8_t *levels, int C);

// the P4's vector decoder (mmrt_p4.S): the table transposed to [level][lane] first. Its 64-bit
// loads need the index rows 8-byte aligned (unaligned, they silently read the wrong bytes):
// otherwise they're copied, 64 rows at a time.
static void cb4_group_p4(int8_t *dst, const uint8_t *src, int C)
{
    int8_t levels[256] __attribute__((aligned(16)));
    for (int k = 0; k < 16; k++)
        for (int i = 0; i < 16; i++) levels[k * 16 + i] = (int8_t)src[i * 16 + k];
    const uint8_t *idx = src + 256;
    if (!((uintptr_t)idx & 7)) {
        mmrt_p4_cb4_rows(dst, idx, levels, C);
        return;
    }
    uint8_t rows[64 * 8] __attribute__((aligned(16)));
    for (int c = 0; c < C; c += 64) {
        const int n = C - c < 64 ? C - c : 64;
        memcpy(rows, idx + (size_t)c * 8, (size_t)n * 8);
        mmrt_p4_cb4_rows(dst + (size_t)c * 16, rows, levels, n);
    }
}
#endif

void mmrt_cb4_group_port(int8_t *dst, const uint8_t *src, int C, uint16_t *pt)
{
#ifdef MMRT_P4
    if (C > 0 && !((uintptr_t)dst & 15) && cb4_p4_ok()) {
        cb4_group_p4(dst, src, C);
        return;
    }
#endif
    const uint8_t *idx = src + 256;
    if (!pt || C < 1024) {            // look each weight up in its lane's table: a row of 16 built
        uint8_t tab[256];             // in registers, stored as four words (tables cost more than
        memcpy(tab, src, 256);        // they save below ~1024 inputs: measured on the P4)
        for (int c = 0; c < C; c++, idx += 8, dst += 16) {
            uint32_t row[4];
            for (int q = 0; q < 4; q++) {
                const uint8_t b0 = idx[2 * q], b1 = idx[2 * q + 1];
                const uint8_t *t = tab + q * 64;   // lanes 4q .. 4q + 3
                row[q] = (uint32_t)t[b0 & 15] | (uint32_t)t[16 + (b0 >> 4)] << 8
                         | (uint32_t)t[32 + (b1 & 15)] << 16 | (uint32_t)t[48 + (b1 >> 4)] << 24;
            }
            memcpy(dst, row, 16);
        }
        return;
    }
    for (int j = 0; j < 8; j++) {     // lanes 2j (low nibble) and 2j + 1 (high): every index byte
        const uint8_t *lo = src + (2 * j) * 16, *hi = src + (2 * j + 1) * 16;
        uint16_t *t = pt + j * 256;
        for (int h = 0; h < 16; h++)
            for (int l = 0; l < 16; l++) t[h * 16 + l] = (uint16_t)(lo[l] | (hi[h] << 8));
    }
    for (int c = 0; c < C; c++, idx += 8, dst += 16) {
        uint8_t pair[16];
        for (int j = 0; j < 8; j++) {
            const uint16_t v = pt[j * 256 + idx[j]];
            pair[2 * j] = (uint8_t)v;
            pair[2 * j + 1] = (uint8_t)(v >> 8);
        }
        memcpy(dst, pair, 16);
    }
}

void mmrt_repack_nc(int8_t *w_nc, const int8_t *w, int C, int N)
{
    for (int g = 0; g < N / 16; g++)
        for (int c = 0; c < C; c++) {
            const int8_t *src = w + ((size_t)g * C + c) * 16;
            for (int j = 0; j < 16; j++) w_nc[(size_t)(g * 16 + j) * C + c] = src[j];
        }
}

// frames [t0, t1) in tiles of TILE_T: each output's row of weights meets a tile's frames while
// both are in cache (TILE_T x C: 8 KB at C 256), so the weights (in PSRAM) are read once per tile
// and the frames (in PSRAM too, once a long clip outgrows internal RAM) once per tile -- rather
// than every frame once per output (8 s of speech took 30 s that way)
#define TILE_T 32
static void conv1x1_nc_frames(const int8_t *x, int C, const int8_t *w_nc, const int32_t *bias, int N,
                              int stride, int shift, int relu, int8_t *y, int t0, int t1)
{
    for (int tb = t0; tb < t1; tb += TILE_T) {
        const int te = tb + TILE_T < t1 ? tb + TILE_T : t1;
        for (int n = 0; n < N; n++) {
            const int8_t *wn = w_nc + (size_t)n * C;
            const int32_t b = bias ? bias[n] : 0;
            for (int t = tb; t < te; t++)
                y[(size_t)t * N + n] = out8(mmrt_dot_s8(x + (size_t)t * stride * C, wn, C / 16) + b, shift, relu);
        }
    }
}

#ifdef MMRT_P4
// ---- both cores: a helper task on core 0 takes the first half of the frames, the caller the
// second (as MMRT's S3 kernels split their work, mmrt_s3.c run_parts); each writes its own
// rows of the output, so the result is the same as one core's
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

// the second core's helper: near the top, unless speech is to yield to the VM (a screen board:
// CONFIG_BABYTALK_YIELD -- the AtomVM schedulers' priority, taking turns with them)
#ifdef CONFIG_BABYTALK_YIELD
#define MMRT_HELPER_PRIORITY 1
#else
#define MMRT_HELPER_PRIORITY (configMAX_PRIORITIES - 2)
#endif

typedef void (*part_fn)(void *arg, int t0, int t1);
static part_fn s_fn;
static void *s_arg;
static int s_t0, s_t1;
static SemaphoreHandle_t s_go, s_done;

static void helper(void *unused)
{
    (void)unused;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        s_fn(s_arg, s_t0, s_t1);
        xSemaphoreGive(s_done);
    }
}

// fn over frames [0, T): half on each core (all on this one if the helper can't start, or T < 2)
static void split_frames(part_fn fn, void *arg, int T)
{
    if (!s_go && T >= 2) {
        s_go = xSemaphoreCreateBinary();
        s_done = xSemaphoreCreateBinary();
        if (!s_go || !s_done || xTaskCreatePinnedToCore(helper, "mmrt_w0", 4096, NULL, MMRT_HELPER_PRIORITY, NULL, 0) != pdPASS)
            s_go = NULL;
    }
    if (!s_go || T < 2 || xPortGetCoreID() == 0) {
        fn(arg, 0, T);
        return;
    }
    s_fn = fn;
    s_arg = arg;
    s_t0 = 0;
    s_t1 = T / 2;
    xSemaphoreGive(s_go);
    fn(arg, T / 2, T);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

typedef struct {
    const int8_t *x, *w_nc;
    const int32_t *bias;
    int8_t *y;
    int C, N, stride, shift, relu;
} c1_job_t;

static void c1_part(void *arg, int t0, int t1)
{
    const c1_job_t *j = (const c1_job_t *)arg;
    conv1x1_nc_frames(j->x, j->C, j->w_nc, j->bias, j->N, j->stride, j->shift, j->relu, j->y, t0, t1);
}
#endif

void mmrt_conv1x1_nc(const int8_t *x, int C, const int8_t *w_nc, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out)
{
#ifdef MMRT_P4
    c1_job_t j = {x, w_nc, bias, y, C, N, stride, shift, relu};
    split_frames(c1_part, &j, T_out);
#else
    conv1x1_nc_frames(x, C, w_nc, bias, N, stride, shift, relu, y, 0, T_out);
#endif
}

#ifndef MMRT_P4  // the P4's is esp.vmulas.s8.xacc (mmrt_p4.S)
int32_t mmrt_dot_s8(const int8_t *a, const int8_t *b, int n16)
{
    int32_t acc = 0;
    for (int i = 0; i < 16 * n16; i++) acc += a[i] * b[i];
    return acc;
}
#endif

#ifdef MMRT_P4
// ---- the depthwise conv on the P4's vector unit (mmrt_p4.S), checked against the portable
// kernel once before its first use
#include <stdio.h>
#include <string.h>

typedef struct {
    int nk, C, groups, K16, shift, relu, pad_[2];
    int8_t rnd_a[16], rnd_b[16];   // rnd_a[i] * rnd_b[0] = 2^(shift - 1): the rounding term
} __attribute__((aligned(16))) mmrt_p4_dw_t;
void mmrt_p4_dw_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_p4_dw_t *a);

// the rounding term 2^(shift - 1) as a product of two int8s, rnd_a[i] * rnd_b[0] (shift 1..13)
static void rnd_factors(int8_t rnd_a[16], int8_t rnd_b[16], int shift)
{
    memset(rnd_a, 0, 16);
    memset(rnd_b, 0, 16);
    if (shift <= 0) return;
    const int r = 1 << (shift - 1), A = r < 64 ? r : 64;
    memset(rnd_a, A, 16);
    rnd_b[0] = (int8_t)(r / A);
}

static void dw_p4(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad, int shift,
                  int relu, int8_t *y, int T_out)
{
    mmrt_p4_dw_t a = {0, C, C / 16, K * 16, shift, relu, {0, 0}, {0}, {0}};
    rnd_factors(a.rnd_a, a.rnd_b, shift);
    for (int t = 0; t < T_out; t++) {
        const int start = t * stride - pad;              // input frame of tap 0
        const int k0 = start < 0 ? -start : 0, k1 = start + K > T_in ? T_in - start : K;
        a.nk = k1 - k0;
        int8_t *yt = y + (size_t)t * C;
        if (a.nk <= 0) {                                 // every tap in the padding
            memset(yt, out8(0, shift, relu), (size_t)C);
            continue;
        }
        mmrt_p4_dw_row(yt, x + (size_t)(start + k0) * C, w + k0 * 16, &a);
    }
}

// 1: the P4 kernel matched the portable one bit for bit on a test layer; 0: it didn't (never used)
static int s_dw_p4_ok = -1;

static int dw_p4_check(void)
{
    enum { C = 48, T = 24, KMAX = 13 };
    static int8_t x[T * C] __attribute__((aligned(16))), w[C * KMAX] __attribute__((aligned(16))),
        y1[T * C] __attribute__((aligned(16))), y2[T * C] __attribute__((aligned(16)));
    uint32_t r = 12345;
    for (int trial = 0; trial < 60; trial++) {
        for (int i = 0; i < T * C; i++) x[i] = (int8_t)((r = r * 1103515245 + 12345) >> 16);
        for (int i = 0; i < C * KMAX; i++) w[i] = (int8_t)((r = r * 1103515245 + 12345) >> 16);
        const int K = 1 + trial % KMAX, stride = 1 + trial % 2, pad = (K - 1) / 2, shift = trial % 14,
                  relu = trial & 1, Ty = (T + 2 * pad - K) / stride + 1;
        mmrt_dwconv_port(x, T, C, w, K, stride, pad, shift, relu, y1, Ty);
        dw_p4(x, T, C, w, K, stride, pad, shift, relu, y2, Ty);
        if (memcmp(y1, y2, (size_t)Ty * C)) {
            printf("mmrt: the P4 depthwise kernel differs from the portable one (K %d shift %d): not used\n", K, shift);
            return 0;
        }
    }
    return 1;
}

// the vector dot product against C's, lengths 1..24 x 16, once before the first row-dot layer
int mmrt_dot_p4_check(void)
{
    static int8_t a[24 * 16] __attribute__((aligned(16))), b[24 * 16] __attribute__((aligned(16)));
    uint32_t r = 777;
    for (int i = 0; i < 24 * 16; i++) {
        a[i] = (int8_t)((r = r * 1103515245 + 12345) >> 16);
        b[i] = (int8_t)((r = r * 1103515245 + 12345) >> 16);
    }
    for (int n16 = 1; n16 <= 24; n16++) {
        int32_t want = 0;
        for (int i = 0; i < 16 * n16; i++) want += a[i] * b[i];
        const int32_t got = mmrt_dot_s8(a, b, n16);
        if (got != want) {
            printf("mmrt: the P4 dot product differs (%d x 16: %ld, not %ld): row dots not used\n", n16, (long)got, (long)want);
            return 0;
        }
    }
    return 1;
}

int mmrt_dwconv_p4(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad, int shift,
                   int relu, int8_t *y, int T_out)
{
    if (s_dw_p4_ok < 0) s_dw_p4_ok = dw_p4_check();
    if (!s_dw_p4_ok || shift < 0 || shift > 13 || C % 16 || ((uintptr_t)x | (uintptr_t)w | (uintptr_t)y) & 15)
        return -1;
    dw_p4(x, T_in, C, w, K, stride, pad, shift, relu, y, T_out);
    return 0;
}
#endif

#if defined(MMRT_P4) || defined(MMRT_C1)
// ---- 1x1 convs / matmuls one 16-output group at a time, from the packed [C][16] layout: on the
// P4 its vector unit (mmrt_p4_c1_group: 16 weights times one input per instruction, into 20-bit
// accumulator lanes), elsewhere (MMRT_C1: the PC checker) the same arithmetic in C. A bias rides
// as one more 16-input chunk: constant inputs X_BIAS = {1, 64 x 15} and a block of weights
// (mmrt_c1_bias_block) that they turn into each lane's bias, exactly.

static const int8_t X_BIAS[16] = {1, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};

int mmrt_c1_bias_block(int8_t blk[256], const int32_t *bias)
{
    memset(blk, 0, 256);
    for (int i = 0; i < 16; i++) {
        const int32_t b = bias[i], w0 = ((b % 64) + 64) % 64;   // bias = w0 + 64 * rest
        int32_t rest = (b - w0) / 64;
        blk[i] = (int8_t)w0;
        for (int k = 1; k < 16 && rest; k++) {
            const int32_t d = rest > 127 ? 127 : (rest < -127 ? -127 : rest);
            blk[k * 16 + i] = (int8_t)d;
            rest -= d;
        }
        if (rest) return -1;                                     // |bias| over 64 * 15 * 127
    }
    return 0;
}

#ifdef MMRT_P4
typedef struct {
    int T, C, y_stride, shift, relu, x_skip;
    const int8_t *bias;
    int pad_;
    int8_t rnd_a[16], rnd_b[16], x_bias[16];
} __attribute__((aligned(16))) mmrt_p4_c1_t;
void mmrt_p4_c1_group(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_p4_c1_t *a);
#endif

// one group over T frames: y[t][i] (rows y_stride apart) from x rows C + x_skip apart
static void c1_group(int8_t *y, const int8_t *x, const int8_t *w, const int8_t *bias_blk, int T, int C,
                     int x_skip, int y_stride, int shift, int relu)
{
#ifdef MMRT_P4
    mmrt_p4_c1_t a = {T, C, y_stride, shift, relu, x_skip, bias_blk, 0, {0}, {0}, {0}};
    rnd_factors(a.rnd_a, a.rnd_b, shift);
    memcpy(a.x_bias, X_BIAS, 16);
    mmrt_p4_c1_group(y, x, w, &a);
#else
    for (int t = 0; t < T; t++, x += C + x_skip, y += y_stride)
        for (int i = 0; i < 16; i++) {
            int32_t acc = 0;
            for (int c = 0; c < C; c++) acc += x[c] * w[c * 16 + i];
            if (bias_blk)
                for (int k = 0; k < 16; k++) acc += X_BIAS[k] * bias_blk[k * 16 + i];
            y[i] = out8(acc, shift, relu);
        }
#endif
}

void mmrt_c1_matmul_group(const int8_t *x, int T, int C, const int8_t *w, int shift, int relu, int8_t *y, int y_stride)
{
    c1_group(y, x, w, NULL, T, C, 0, y_stride, shift, relu);
}

typedef struct {
    const int8_t *x, *w, *bias_blks;
    int8_t *y;
    int C, N, stride, shift, relu;
} c1_conv_t;

// frames [t0, t1), in tiles of TILE_T frames (a tile's inputs stay in cache while every group
// of weights meets them)
static void c1_frames(void *arg, int t0, int t1)
{
    const c1_conv_t *j = (const c1_conv_t *)arg;
    for (int tb = t0; tb < t1; tb += TILE_T) {
        const int n = tb + TILE_T < t1 ? TILE_T : t1 - tb;
        for (int g = 0; g < j->N / 16; g++)
            c1_group(j->y + (size_t)tb * j->N + g * 16, j->x + (size_t)tb * j->stride * j->C,
                     j->w + (size_t)g * j->C * 16, j->bias_blks ? j->bias_blks + g * 256 : NULL, n, j->C,
                     (j->stride - 1) * j->C, j->N, j->shift, j->relu);
    }
}

void mmrt_conv1x1_c1(const int8_t *x, int C, const int8_t *w, const int8_t *bias_blks, int N, int stride,
                     int shift, int relu, int8_t *y, int T_out)
{
    c1_conv_t j = {x, w, bias_blks, y, C, N, stride, shift, relu};
#ifdef MMRT_P4
    split_frames(c1_frames, &j, T_out);
#else
    c1_frames(&j, 0, T_out);
#endif
}
#endif

#if defined(MMRT_P4) || defined(MMRT_C1)
// ---- a block's tail: C, and on the P4 its vector unit (mmrt_p4_tail_row)
static void tail_c(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y)
{
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) {
            const size_t k = (size_t)t * C + c;
            int32_t v = a[k];
            if (s) {
                v = (v * s[c] + (1 << (shift - 1))) >> shift;
                v = v > 127 ? 127 : (v < -128 ? -128 : v);
            }
            if (r) {
                v += r[k];
                v = v > 127 ? 127 : (v < -128 ? -128 : v);
            }
            y[k] = (int8_t)(relu && v < 0 ? 0 : v);
        }
}

#ifdef MMRT_P4
typedef struct {
    int groups, shift, relu, pad_;
    int8_t rnd_a[16], rnd_b[16], ones[16];
} __attribute__((aligned(16))) mmrt_p4_tail_t;
void mmrt_p4_tail_row(int8_t *y, const int8_t *a, const int8_t *s, const int8_t *r, const mmrt_p4_tail_t *t);

static void tail_p4(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y)
{
    mmrt_p4_tail_t t = {C / 16, shift, relu, 0, {0}, {0}, {0}};
    rnd_factors(t.rnd_a, t.rnd_b, shift);
    memset(t.ones, 1, 16);
    for (int i = 0; i < T; i++) {
        const size_t off = (size_t)i * C;
        mmrt_p4_tail_row(y + off, a + off, s, r ? r + off : NULL, &t);
    }
}

// the vector tail against C, once: with and without the scale and the residual, every shift
static int tail_p4_ok(void)
{
    static int ok = -1;
    if (ok >= 0) return ok;
    enum { T = 3, C = 48 };
    static int8_t a[T * C] __attribute__((aligned(16))), s[C] __attribute__((aligned(16))),
        r[T * C] __attribute__((aligned(16))), y1[T * C] __attribute__((aligned(16))), y2[T * C] __attribute__((aligned(16)));
    uint32_t q = 31337;
    for (int trial = 0; trial < 52; trial++) {
        for (int i = 0; i < T * C; i++) a[i] = (int8_t)((q = q * 1103515245 + 12345) >> 16);
        for (int i = 0; i < T * C; i++) r[i] = (int8_t)((q = q * 1103515245 + 12345) >> 16);
        for (int i = 0; i < C; i++) s[i] = (int8_t)((q = q * 1103515245 + 12345) >> 16);
        const int shift = 1 + trial % 13, with_s = trial % 4 != 0, with_r = trial % 4 != 1, relu = (trial / 4) & 1;
        tail_c(a, with_s ? s : NULL, with_r ? r : NULL, T, C, shift, relu, y1);
        tail_p4(a, with_s ? s : NULL, with_r ? r : NULL, T, C, shift, relu, y2);
        if (memcmp(y1, y2, sizeof(y1))) {
            printf("mmrt: the P4 tail kernel differs (shift %d, scale %d, residual %d): not used\n", shift, with_s, with_r);
            return ok = 0;
        }
    }
    return ok = 1;
}
#endif

void mmrt_tail_c1(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y)
{
#ifdef MMRT_P4
    if (C % 16 == 0 && shift >= 1 && shift <= 13 && !(((uintptr_t)a | (uintptr_t)s | (uintptr_t)r | (uintptr_t)y) & 15)
        && tail_p4_ok()) {
        tail_p4(a, s, r, T, C, shift, relu, y);
        return;
    }
#endif
    tail_c(a, s, r, T, C, shift, relu, y);
}
#endif

#ifdef MMRT_P4
// the group kernel against the same arithmetic in C, once: biases, strides, every shift
static int s_c1_ok = -1;

int mmrt_p4_matmul_ok(void)
{
    if (s_c1_ok >= 0) return s_c1_ok;
    enum { T = 5, C = 64, STRIDE = 2 };
    static int8_t x[T * STRIDE * C] __attribute__((aligned(16))), w[C * 16] __attribute__((aligned(16))),
        blk[256] __attribute__((aligned(16))), y[T * 32] __attribute__((aligned(16)));
    uint32_t r = 4242;
    for (int trial = 0; trial < 28; trial++) {
        const int range = trial < 14 ? 16 : 45, shift = trial % 14, relu = trial & 1, with_bias = trial % 3 != 0,
                  stride = 1 + (trial & 2) / 2;   // |sum| < 64 * 45 * 45 + 2^16 < 2^19
        int32_t bias[16];
        for (int i = 0; i < T * STRIDE * C; i++) x[i] = (int8_t)((int)((r = r * 1103515245 + 12345) >> 16) % range);
        for (int i = 0; i < C * 16; i++) w[i] = (int8_t)((int)((r = r * 1103515245 + 12345) >> 16) % range);
        for (int i = 0; i < 16; i++) bias[i] = (int32_t)((r = r * 1103515245 + 12345) >> 15) % 65536 - 32768;
        if (mmrt_c1_bias_block(blk, bias)) return s_c1_ok = 0;
        c1_group(y, x, w, with_bias ? blk : NULL, T, C, (stride - 1) * C, 32, shift, relu);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < 16; i++) {
                int32_t acc = with_bias ? bias[i] : 0;
                for (int c = 0; c < C; c++) acc += x[t * stride * C + c] * w[c * 16 + i];
                const int8_t want = out8(acc, shift, relu);
                if (y[t * 32 + i] != want) {
                    printf("mmrt: the P4 1x1 kernel differs (shift %d, bias %d, stride %d: %d, not %d): not used\n",
                           shift, with_bias, stride, y[t * 32 + i], want);
                    return s_c1_ok = 0;
                }
            }
    }
    return s_c1_ok = 1;
}
#endif

#ifdef MMRT_P4
// the vector CB4 decoder against the table lookup, once
static int cb4_p4_ok(void)
{
    static int ok = -1;
    if (ok >= 0) return ok;
    enum { C = 80 };   // more than one 64-row chunk
    static uint8_t buf[256 + C * 8 + 8] __attribute__((aligned(16)));
    static int8_t want[C * 16] __attribute__((aligned(16))), got[C * 16] __attribute__((aligned(16)));
    uint32_t r = 99;
    for (int off = 0; off < 4; off++) {       // the source aligned, and not (0, 1, 3, 5 bytes off)
        const uint8_t *src = buf + (off ? 2 * off - 1 : 0);
        for (int i = 0; i < (int)sizeof(buf); i++) buf[i] = (uint8_t)((r = r * 1103515245 + 12345) >> 16);
        for (int c = 0; c < C; c++)
            for (int j = 0; j < 8; j++) {
                const uint8_t b = src[256 + c * 8 + j];
                want[c * 16 + 2 * j] = (int8_t)src[(2 * j) * 16 + (b & 15)];
                want[c * 16 + 2 * j + 1] = (int8_t)src[(2 * j + 1) * 16 + (b >> 4)];
            }
        cb4_group_p4(got, src, C);
        if (memcmp(want, got, sizeof(want))) {
            printf("mmrt: the P4 int4 decoder differs (source %d bytes off: %d %d %d, not %d %d %d): not used\n",
                   off ? 2 * off - 1 : 0, got[0], got[1], got[2], want[0], want[1], want[2]);
            return ok = 0;
        }
    }
    return ok = 1;
}
#endif
