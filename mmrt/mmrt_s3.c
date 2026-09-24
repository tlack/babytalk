// mmrt ESP32-S3 kernels: C drivers around the PIE assembly in mmrt_s3.S.
#include "mmrt_s3.h"

#include <string.h>

#include "esp_attr.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_memory_utils.h"
#include "esp_heap_caps.h"

typedef struct {
    const uint8_t *bias_q;
    int c16m1;
    int groups;
    int shift;
    int relu;
} mmrt_s3_c1_t;

void mmrt_s3_conv1x1_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_c1_t *a);

typedef struct {
    int nk;
    int C;
    int groups;
    int K16;
    int shift;
    int relu;
} mmrt_s3_dw_t;

void mmrt_s3_dw_row(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_dw_t *a);

typedef struct {
    int nrows, pairs, C, groups, K16, rowstride, shift, relu;
    const uint8_t *rnd;  // 64-byte QACC image: 2^(shift-1) in every lane
    int odd;
} mmrt_s3_dwr_t;

void mmrt_s3_dw_rows(int8_t *y, const int8_t *x, const int8_t *w, const mmrt_s3_dwr_t *a);

typedef struct {
    int groups;
    int shift;
    int relu;
} mmrt_s3_tail_t;

void mmrt_s3_tail_row(int8_t *y, const int8_t *a, const int8_t *s, const int8_t *r, const mmrt_s3_tail_t *t);
void mmrt_s3_colsum16(const int8_t *x, int T, int C, uint8_t *qacc_out);

// Internal-SRAM staging. 1x1 weights are re-read for every output frame, so each op
// copies them out of flash once. Each core gets its own 32KB buffer, in separate SRAM
// regions (a static one and a heap one): two cores streaming weights out of the same
// SRAM bank stall each other (measured: 1.17x -> 1.51x dual-core speedup when split).
#define W_STAGE_BYTES (32 * 1024)
static DRAM_ATTR int8_t s_wstage0[W_STAGE_BYTES] __attribute__((aligned(16)));
static int8_t *s_wstage1;  // heap, allocated on first use
static DRAM_ATTR uint8_t s_bias_q[64 * 48] __attribute__((aligned(16)));  // up to 768 outputs

int mmrt_s3_stage = 1;
int mmrt_s3_cores = 2;

// ---------------------------------------------------------------- two-core split
// A persistent worker pinned to core 0 runs half of each kernel while the caller (the
// inference task on core 1) runs the other half. Work is split by output frames only,
// so results are identical to one core.
typedef void (*part_fn)(void *arg, int part);
static part_fn s_fn;
static void *s_arg;
static SemaphoreHandle_t s_go, s_done;

int64_t mmrt_s3_part_us[2];  // wall time of each core's part in the last split call
// Profiling counters (cumulative, reset by the caller): 1x1 weight staging and compute
// per core (index 0 = worker / single-core, 1 = caller's half).
int64_t mmrt_s3_stage_us[2], mmrt_s3_c1_us[2];

static void timed(part_fn fn, void *arg, int part)
{
    int64_t t0 = esp_timer_get_time();
    fn(arg, part);
    mmrt_s3_part_us[part] = esp_timer_get_time() - t0;
}

static void worker(void *unused)
{
    (void)unused;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        timed(s_fn, s_arg, 0);
        xSemaphoreGive(s_done);
    }
}

static void run_parts(part_fn fn, void *arg)
{
    if (mmrt_s3_cores < 2) {
        fn(arg, -1);  // whole range
        return;
    }
    if (!s_go) {
        s_go = xSemaphoreCreateBinary();
        s_done = xSemaphoreCreateBinary();
        xTaskCreatePinnedToCore(worker, "mmrt_w0", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 0);
    }
    s_fn = fn;
    s_arg = arg;
    xSemaphoreGive(s_go);
    timed(fn, arg, 1);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

// [begin, end) of n items for a part (-1 = all, 0 = first half, 1 = second half)
static void part_range(int n, int part, int *b, int *e)
{
    int h = n / 2;
    *b = part == 1 ? h : 0;
    *e = part == 0 ? h : n;
}

// int32 bias -> QACC load format: per 16-output group, two halves of 8 lanes x 20 bits
// packed little-endian (20 bytes), each half padded to 32 bytes.
static void pack_bias(const int32_t *bias, int N, uint8_t *dst)
{
    memset(dst, 0, (size_t)N / 16 * 64);
    for (int n = 0; n < N; n++) {
        uint8_t *half = dst + (n / 16) * 64 + ((n % 16) / 8) * 32;
        uint32_t v = (uint32_t)bias[n] & 0xfffff;
        int bit = (n % 8) * 20;
        for (int b = 0; b < 20; b++, bit++)
            if (v >> b & 1) half[bit / 8] |= (uint8_t)(1u << (bit % 8));
    }
}

typedef struct {
    mmrt_s3_c1_t a;       // groups/bias_q cover the whole chunk
    int8_t *y;            // output, column offset of the chunk's first group applied
    const int8_t *x, *w;  // w: the chunk's weights (in flash or wherever they live)
    int N, C, stride, T;
} c1_job_t;

// Part p of a chunk: its half of the output groups (all groups for p = -1), staged
// into that core's SRAM buffer, over all frames.
static void c1_part(void *arg, int part)
{
    c1_job_t *j = (c1_job_t *)arg;
    int b, e;
    part_range(j->a.groups, part, &b, &e);
    if (e <= b) return;
    const size_t group_bytes = (size_t)j->C * 16;
    const int8_t *w = j->w + (size_t)b * group_bytes;
    const int core = part == 1 ? 1 : 0;
    int64_t t0 = esp_timer_get_time();
    if (mmrt_s3_stage && !esp_ptr_internal(w)) {
        int8_t *buf = part == 1 ? s_wstage1 : s_wstage0;
        memcpy(buf, w, (size_t)(e - b) * group_bytes);
        w = buf;
    }
    int64_t t1 = esp_timer_get_time();
    mmrt_s3_c1_t a = j->a;
    a.groups = e - b;
    if (a.bias_q) a.bias_q += (size_t)b * 64;
    for (int t = 0; t < j->T; t++)
        mmrt_s3_conv1x1_row(j->y + (size_t)t * j->N + b * 16, j->x + (size_t)t * j->stride * j->C, w, &a);
    mmrt_s3_stage_us[core] += t1 - t0;
    mmrt_s3_c1_us[core] += esp_timer_get_time() - t1;
}

void mmrt_s3_conv1x1(const int8_t *x, int T_in, int C, const int8_t *w, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out)
{
    (void)T_in;
    if (!s_wstage1) s_wstage1 = (int8_t *)heap_caps_aligned_alloc(16, W_STAGE_BYTES, MALLOC_CAP_INTERNAL);
    const int two = mmrt_s3_cores == 2 && s_wstage1;
    const int groups = N / 16;
    const size_t group_bytes = (size_t)C * 16;
    int per_buf = (int)(W_STAGE_BYTES / group_bytes);  // groups one staging buffer holds
    int chunk = two ? 2 * per_buf : per_buf;
    if (chunk > groups) chunk = groups;
    if (bias) pack_bias(bias, N, s_bias_q);
    for (int g0 = 0; g0 < groups; g0 += chunk) {
        int ng = groups - g0 < chunk ? groups - g0 : chunk;
        c1_job_t job = {{bias ? s_bias_q + (size_t)g0 * 64 : NULL, C / 16 - 1, ng, shift, relu},
                        y + g0 * 16, x, w + (size_t)g0 * group_bytes, N, C, stride, T_out};
        if (two && ng > 1)
            run_parts(c1_part, &job);
        else
            c1_part(&job, -1);
    }
}

typedef struct {
    const int8_t *x, *w;
    int8_t *y;
    int T_in, C, K, stride, pad, shift, relu, T_out;
} dw_job_t;

int mmrt_s3_dw_fast = 1;  // 1: interior frames via mmrt_s3_dw_rows

static void dw_part(void *arg, int part)
{
    dw_job_t *j = (dw_job_t *)arg;
    mmrt_s3_dw_t a = {0, j->C, j->C / 16, j->K * 16, j->shift, j->relu};
    int b, e;
    part_range(j->T_out, part, &b, &e);
    // interior frames: every tap inside the clip
    int in0 = (j->pad + j->stride - 1) / j->stride;                   // first t with start >= 0
    int in1 = j->T_in - j->K + j->pad >= 0 ? (j->T_in - j->K + j->pad) / j->stride + 1 : 0;  // first t past the end
    if (in0 < b) in0 = b;
    if (in1 > e) in1 = e;
    if (!mmrt_s3_dw_fast || in1 <= in0 || j->K < 3) in0 = in1 = e;
    for (int t = b; t < e; t++) {
        if (t == in0) {
            static DRAM_ATTR uint8_t rnd_core[2][64] __attribute__((aligned(16)));
            uint8_t *rnd = rnd_core[part == 1];
            int32_t r[16];
            for (int i = 0; i < 16; i++) r[i] = j->shift > 0 ? 1 << (j->shift - 1) : 0;
            pack_bias(r, 16, rnd);
            int odd = j->K & 1, pairs = odd ? (j->K - 3) / 2 : (j->K - 2) / 2;
            mmrt_s3_dwr_t ra = {in1 - in0, pairs, j->C, j->C / 16, j->K * 16, j->stride * j->C, j->shift, j->relu, rnd, odd};
            mmrt_s3_dw_rows(j->y + (size_t)t * j->C, j->x + (size_t)(t * j->stride - j->pad) * j->C, j->w, &ra);
            t = in1 - 1;
            continue;
        }
        int start = t * j->stride - j->pad;  // input frame of tap 0
        int k0 = start < 0 ? -start : 0;
        int k1 = start + j->K > j->T_in ? j->T_in - start : j->K;
        a.nk = k1 - k0;
        mmrt_s3_dw_row(j->y + (size_t)t * j->C, j->x + (size_t)(start + k0) * j->C, j->w + k0 * 16, &a);
    }
}

void mmrt_s3_dwconv(const int8_t *x, int T_in, int C, const int8_t *w, int K, int stride, int pad,
                    int shift, int relu, int8_t *y, int T_out)
{
    dw_job_t job = {x, w, y, T_in, C, K, stride, pad, shift, relu, T_out};
    run_parts(dw_part, &job);
}

void mmrt_s3_tail(const int8_t *a, const int8_t *s, const int8_t *r, int T, int C, int shift, int relu, int8_t *y)
{
    mmrt_s3_tail_t t = {C / 16, shift, relu};
    for (int i = 0; i < T; i++) {
        size_t off = (size_t)i * C;
        mmrt_s3_tail_row(y + off, a + off, s, r ? r + off : NULL, &t);
    }
}

// Unpack one 64-byte QACC dump (see pack_bias) into 16 sign-extended int32 lanes.
static void unpack_qacc(const uint8_t *q, int32_t *out)
{
    for (int n = 0; n < 16; n++) {
        const uint8_t *half = q + (n / 8) * 32;
        int bit = (n % 8) * 20;
        uint32_t v = 0;
        for (int b = 0; b < 20; b++, bit++) v |= (uint32_t)(half[bit / 8] >> (bit % 8) & 1) << b;
        out[n] = (int32_t)(v << 12) >> 12;
    }
}

static inline int64_t floordiv64(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

void mmrt_s3_mean(const int8_t *x, int T, int C, int e_in, int e_out, int8_t *y)
{
    static DRAM_ATTR uint8_t q[64] __attribute__((aligned(16)));
    int32_t sums[16];
    int k = e_in - e_out;  // value = sum * 2^k / T, rounded half up (as mmrt_mean_ref)
    for (int g = 0; g < C / 16; g++) {
        mmrt_s3_colsum16(x + g * 16, T, C, q);
        unpack_qacc(q, sums);
        for (int i = 0; i < 16; i++) {
            int64_t num = k >= 0 ? (int64_t)sums[i] << k : sums[i], den = k >= 0 ? T : (int64_t)T << -k;
            int64_t v = floordiv64(2 * num + den, 2 * den);
            y[g * 16 + i] = v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v);
        }
    }
}
