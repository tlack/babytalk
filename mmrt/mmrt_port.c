#include "mmrt_port.h"

#include <stddef.h>

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
        if (!s_go || !s_done || xTaskCreatePinnedToCore(helper, "mmrt_w0", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 0) != pdPASS)
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

static void dw_p4(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad, int shift,
                  int relu, int8_t *y, int T_out)
{
    mmrt_p4_dw_t a = {0, C, C / 16, K * 16, shift, relu, {0, 0}, {0}, {0}};
    if (shift > 0) {                                     // 2^(shift - 1) as a product of two int8s
        const int r = 1 << (shift - 1), A = r < 64 ? r : 64;
        memset(a.rnd_a, A, sizeof(a.rnd_a));
        a.rnd_b[0] = (int8_t)(r / A);
    }
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
