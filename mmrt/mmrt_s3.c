// mmrt ESP32-S3 kernels: C drivers around the PIE assembly in mmrt_s3.S.
#include "mmrt_s3.h"

#include "mmrt.h"

#include <string.h>

#include "esp_attr.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_memory_utils.h"
#include "esp_heap_caps.h"
#include "esp_cpu.h"

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
static int8_t *s_wstage1;  // heap: mmrt_s3_init(), else on first use
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

static void ensure_worker(void)
{
    if (!s_go) {
        s_go = xSemaphoreCreateBinary();
        s_done = xSemaphoreCreateBinary();
        xTaskCreatePinnedToCore(worker, "mmrt_w0", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 0);
    }
}

int mmrt_s3_init(void)
{
    if (!s_wstage1) s_wstage1 = (int8_t *)heap_caps_aligned_alloc(16, W_STAGE_BYTES, MALLOC_CAP_INTERNAL);
    ensure_worker();
    return s_wstage1 && s_go ? 0 : -1;
}

// ---------------------------------------------------------------- weight streaming
// Short inputs: 1x1 weights must come out of flash (~310 ms per inference, flash-bus
// bound -- two cores copy no faster than one) but compute is small. So core 0 becomes a
// streamer that copies the upcoming 1x1 layers' weights, whole output-channel groups at a
// time, through a ring of four 16KB SRAM slots (the same 64KB as the staging buffers),
// while core 1 computes everything else and consumes the slots in order. Flash reads
// then overlap compute instead of alternating with it.
#define SLOT_BYTES (16 * 1024)
#define NSLOT 4
static struct {
    const mmrt_s3_stream_op_t *ops;
    int n_ops;
    SemaphoreHandle_t free_slots, full_slots;
    int8_t *slot[NSLOT];
    struct { int op, g0, ng; } desc[NSLOT];
    volatile int stop;
    int consumed;       // slots taken by the consumer
    int op_seq;         // 1x1 ops started by the consumer
    int active;
} S;

int mmrt_s3_stream_max_T = 250;  // input frames (2.5 s); measured crossover ~2.7 s. 0 disables

static int slot_groups(int C)
{
    int g = SLOT_BYTES / (C * 16);
    return g < 1 ? 1 : g;
}

static void streamer(void *arg, int part)
{
    (void)arg;
    (void)part;
    int s = 0;
    for (int i = 0; i < S.n_ops && !S.stop; i++) {
        const mmrt_s3_stream_op_t *op = &S.ops[i];
        const int groups = op->N / 16, cg = slot_groups(op->C);
        for (int g0 = 0; g0 < groups; g0 += cg) {
            int ng = groups - g0 < cg ? groups - g0 : cg;
            xSemaphoreTake(S.free_slots, portMAX_DELAY);
            if (S.stop) return;
            int64_t t0 = esp_timer_get_time();
            mmrt_wload(S.slot[s], op->w, op->wfmt, op->C, g0, ng);
            mmrt_s3_stage_us[0] += esp_timer_get_time() - t0;
            S.desc[s].op = i;
            S.desc[s].g0 = g0;
            S.desc[s].ng = ng;
            xSemaphoreGive(S.full_slots);
            s = (s + 1) % NSLOT;
        }
    }
}

int mmrt_s3_stream_begin(const mmrt_s3_stream_op_t *ops, int n_ops, int T_in)
{
    if (!mmrt_s3_stage || mmrt_s3_cores < 2 || T_in > mmrt_s3_stream_max_T || n_ops == 0) return 0;
    if (!s_wstage1) s_wstage1 = (int8_t *)heap_caps_aligned_alloc(16, W_STAGE_BYTES, MALLOC_CAP_INTERNAL);
    if (!s_wstage1) return 0;
    for (int i = 0; i < n_ops; i++)
        if (ops[i].C * 16 > SLOT_BYTES || ops[i].N % 16) return 0;
    ensure_worker();
    if (!S.free_slots) {
        S.free_slots = xSemaphoreCreateCounting(NSLOT, NSLOT);
        S.full_slots = xSemaphoreCreateCounting(NSLOT, 0);
    }
    S.slot[0] = s_wstage0;
    S.slot[1] = s_wstage0 + SLOT_BYTES;
    S.slot[2] = s_wstage1;
    S.slot[3] = s_wstage1 + SLOT_BYTES;
    S.ops = ops;
    S.n_ops = n_ops;
    S.stop = 0;
    S.consumed = 0;
    S.op_seq = 0;
    S.active = 1;
    s_fn = streamer;
    s_arg = NULL;
    xSemaphoreGive(s_go);  // the worker runs the streamer until the list is done
    return 1;
}

void mmrt_s3_stream_end(void)
{
    if (!S.active) return;
    S.stop = 1;
    for (int i = 0; i < NSLOT; i++) xSemaphoreGive(S.free_slots);  // unblock the streamer
    xSemaphoreTake(s_done, portMAX_DELAY);
    while (xSemaphoreTake(S.full_slots, 0) == pdTRUE) {}      // reset both counts
    while (xSemaphoreTake(S.free_slots, 0) == pdTRUE) {}
    for (int i = 0; i < NSLOT; i++) xSemaphoreGive(S.free_slots);
    S.active = 0;
}

static void run_parts(part_fn fn, void *arg)
{
    if (mmrt_s3_cores < 2 || S.active) {  // streaming: core 0 is busy copying weights
        fn(arg, -1);  // whole range
        return;
    }
    ensure_worker();
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
    const int8_t *x, *w;  // w: the op's weights (in flash or wherever they live), format wfmt
    int wfmt, g0;         // g0: the chunk's first group
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
    const int8_t *w = j->w + (size_t)(j->g0 + b) * j->C * 16;  // in place (INT8 only)
    const int core = part == 1 ? 1 : 0;
    int64_t t0 = esp_timer_get_time();
    if (j->wfmt != MMRT_W_INT8 || (mmrt_s3_stage && !esp_ptr_internal(w))) {
        int8_t *buf = part == 1 ? s_wstage1 : s_wstage0;
        mmrt_wload(buf, j->w, j->wfmt, j->C, j->g0 + b, e - b);
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

void mmrt_s3_conv1x1(const int8_t *x, int T_in, int C, const int8_t *w, int wfmt, const int32_t *bias, int N,
                     int stride, int shift, int relu, int8_t *y, int T_out)
{
    (void)T_in;
    if (S.active) {  // consume this layer's weights from the streamer's slots, in order
        const int groups = N / 16;
        if (bias) pack_bias(bias, N, s_bias_q);
        for (int g0 = 0; g0 < groups;) {
            xSemaphoreTake(S.full_slots, portMAX_DELAY);
            int si = S.consumed % NSLOT;
            int ng = S.desc[si].ng;
            if (S.desc[si].op != S.op_seq || S.desc[si].g0 != g0) abort();  // executor/streamer out of step
            int64_t t0 = esp_timer_get_time();
            mmrt_s3_c1_t a = {bias ? s_bias_q + (size_t)g0 * 64 : NULL, C / 16 - 1, ng, shift, relu};
            for (int t = 0; t < T_out; t++)
                mmrt_s3_conv1x1_row(y + (size_t)t * N + g0 * 16, x + (size_t)t * stride * C, S.slot[si], &a);
            mmrt_s3_c1_us[1] += esp_timer_get_time() - t0;
            S.consumed++;
            xSemaphoreGive(S.free_slots);
            g0 += ng;
        }
        S.op_seq++;
        return;
    }
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
                        y + g0 * 16, x, w, wfmt, g0, N, C, stride, T_out};
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

// Depthwise output frames [b, e) written to ydst (frame b first, rows C bytes apart):
// the tensor itself, or a fused op's SRAM tile.
//
// (A frame-loop-in-assembly variant with two-tap software pipelining, mmrt_s3_dw_rows,
// was tried and removed: no faster end to end -- depthwise is bus-bound -- and on two
// cores it rarely (~1e-3 per call) produced one wrong 32-bit word in an output row. The
// same loop with a nop before the loop end, or unpipelined, was clean; see
// mmrt/README.md and the `dwstress` command.)
static void dw_frames(const dw_job_t *j, int b, int e, int8_t *ydst)
{
    mmrt_s3_dw_t a = {0, j->C, j->C / 16, j->K * 16, j->shift, j->relu};
    for (int t = b; t < e; t++) {
        int start = t * j->stride - j->pad;  // input frame of tap 0
        int k0 = start < 0 ? -start : 0;
        int k1 = start + j->K > j->T_in ? j->T_in - start : j->K;
        a.nk = k1 - k0;
        mmrt_s3_dw_row(ydst + (size_t)(t - b) * j->C, j->x + (size_t)(start + k0) * j->C, j->w + k0 * 16, &a);
    }
}

static void dw_part(void *arg, int part)
{
    dw_job_t *j = (dw_job_t *)arg;
    int b, e;
    part_range(j->T_out, part, &b, &e);
    dw_frames(j, b, e, j->y + (size_t)b * j->C);
}

// ---------------------------------------------------------------- fused depthwise -> 1x1
// The depthwise output never touches PSRAM: each core computes it 16 frames at a time
// into its own SRAM tile and runs the 1x1 over the tile straight away. Short inputs are
// bus-bound, and this removes a write + read of a [T][C] tensor per pair of ops.
// The 1x1 weights are staged as two halves in the two separate-bank buffers; core 0
// runs half A then B per tile, core 1 B then A, so they tend to read different banks.
#define TILE 16
#define TILE_C 256
static DRAM_ATTR int8_t s_tile[2][TILE * TILE_C] __attribute__((aligned(16)));

typedef struct {
    dw_job_t dw;         // dw.y unused
    int8_t *y;           // 1x1 output [T][N]
    int N, T;
    const int8_t *wA, *wB;
    mmrt_s3_c1_t aA, aB;  // groups [0, aA.groups) in wA, the rest in wB
} fused_job_t;

static void fused_part(void *arg, int part)
{
    fused_job_t *j = (fused_job_t *)arg;
    const int core = part == 1, C = j->dw.C;
    int b, e;
    part_range(j->T, part, &b, &e);
    int8_t *tile = s_tile[core];
    for (int t0 = b; t0 < e; t0 += TILE) {
        int t1 = t0 + TILE < e ? t0 + TILE : e;
        dw_frames(&j->dw, t0, t1, tile);
        for (int pass = 0; pass < 2; pass++) {
            int useB = pass ^ core;
            const mmrt_s3_c1_t *a = useB ? &j->aB : &j->aA;
            if (!a->groups) continue;
            const int8_t *w = useB ? j->wB : j->wA;
            int col = useB ? j->aA.groups * 16 : 0;
            for (int t = t0; t < t1; t++)
                mmrt_s3_conv1x1_row(j->y + (size_t)t * j->N + col, tile + (size_t)(t - t0) * C, w, a);
        }
    }
}

int mmrt_s3_fuse = 1;

int mmrt_s3_dw_pw(const int8_t *x, int T_in, int C, const int8_t *dw_w, int K, int stride, int pad, int dw_shift,
                  int dw_relu, const int8_t *pw_w, int pw_wfmt, const int32_t *bias, int N, int pw_shift, int pw_relu, int8_t *y,
                  int T)
{
    const int G = N / 16, gA = G / 2;
    const size_t gb = (size_t)C * 16;
    if (!mmrt_s3_fuse || S.active || C > TILE_C || (size_t)(G - gA) * gb > W_STAGE_BYTES ||
        (size_t)gA * gb > W_STAGE_BYTES || K < 3)
        return 0;
    if (!s_wstage1) s_wstage1 = (int8_t *)heap_caps_aligned_alloc(16, W_STAGE_BYTES, MALLOC_CAP_INTERNAL);
    if (!s_wstage1) return 0;
    int64_t t0 = esp_timer_get_time();
    mmrt_wload(s_wstage0, pw_w, pw_wfmt, C, 0, gA);
    mmrt_wload(s_wstage1, pw_w, pw_wfmt, C, gA, G - gA);
    mmrt_s3_stage_us[0] += esp_timer_get_time() - t0;
    if (bias) pack_bias(bias, N, s_bias_q);
    fused_job_t j = {{x, dw_w, NULL, T_in, C, K, stride, pad, dw_shift, dw_relu, T},
                     y, N, T, s_wstage0, s_wstage1,
                     {bias ? s_bias_q : NULL, C / 16 - 1, gA, pw_shift, pw_relu},
                     {bias ? s_bias_q + (size_t)gA * 64 : NULL, C / 16 - 1, G - gA, pw_shift, pw_relu}};
    run_parts(fused_part, &j);
    return 1;
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

// ---------------------------------------------------------------- CB4 decode
void mmrt_s3_cb4_rows(int8_t *dst, const uint8_t *idx, const uint32_t *tab32, int C);

// One CB4 group (256-byte table + C*8 index bytes) -> [C][16] int8, via the asm row decoder.
void mmrt_s3_cb4_group(int8_t *dst, const uint8_t *src, int C)
{
    uint32_t tab32[256];  // [lane][level], each value pre-shifted to its byte of the output word
    for (int i = 0; i < 256; i++) tab32[i] = (uint32_t)src[i] << (8 * ((i >> 4) & 3));
    mmrt_s3_cb4_rows(dst, src + 256, tab32, C);
}

// ---------------------------------------------------------------- kernel microbenchmark
// CPU cycles per VSMULAS (16 MACs) of the 1x1 row kernel on this core: groups x C
// weights and `frames` input rows, each either in internal SRAM or PSRAM (bit 0: x in
// PSRAM, bit 1: w in PSRAM). The ideal is 1.0.
float mmrt_s3_c1_bench(int groups, int C, int frames, int where)
{
    int8_t *x = heap_caps_aligned_alloc(16, (size_t)frames * C, (where & 1) ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL);
    int8_t *w = heap_caps_aligned_alloc(16, (size_t)groups * C * 16, (where & 2) ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL);
    int8_t *y = heap_caps_aligned_alloc(16, (size_t)groups * 16, MALLOC_CAP_INTERNAL);
    float r = -1;
    if (x && w && y) {
        memset(x, 3, (size_t)frames * C);
        memset(w, 5, (size_t)groups * C * 16);
        mmrt_s3_c1_t a = {NULL, C / 16 - 1, groups, 8, 0};
        mmrt_s3_conv1x1_row(y, x, w, &a);  // warm up
        uint32_t c0 = esp_cpu_get_cycle_count();
        for (int t = 0; t < frames; t++) mmrt_s3_conv1x1_row(y, x + (size_t)t * C, w, &a);
        uint32_t c1 = esp_cpu_get_cycle_count();
        r = (float)(c1 - c0) / ((float)frames * groups * C);
    }
    heap_caps_free(x);
    heap_caps_free(w);
    heap_caps_free(y);
    return r;
}

// CB4 decode check + speed, SRAM to SRAM (4 groups, C inputs): cycles per weight of the
// asm decoder (*asm) and the portable C one (*ref); returns the number of differing bytes.
int mmrt_s3_cb4_bench(int C, float *asm_cyc, float *ref_cyc)
{
    const size_t gb = mmrt_wgroup_bytes(C, MMRT_W_CB4), n = (size_t)C * 16 * 4;
    uint8_t *src = heap_caps_aligned_alloc(16, gb * 4, MALLOC_CAP_INTERNAL);
    int8_t *d0 = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
    int8_t *d1 = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
    int bad = -1;
    if (src && d0 && d1) {
        for (size_t i = 0; i < gb * 4; i++) src[i] = (uint8_t)(i * 37 + (i >> 7) * 11);
        extern int mmrt_use_ref;
        int save = mmrt_use_ref;
        mmrt_use_ref = 0;
        mmrt_wload(d0, (const int8_t *)src, MMRT_W_CB4, C, 0, 1);  // warm
        uint32_t c0 = esp_cpu_get_cycle_count();
        mmrt_wload(d0, (const int8_t *)src, MMRT_W_CB4, C, 0, 4);
        *asm_cyc = (float)(esp_cpu_get_cycle_count() - c0) / n;
        mmrt_use_ref = 1;
        c0 = esp_cpu_get_cycle_count();
        mmrt_wload(d1, (const int8_t *)src, MMRT_W_CB4, C, 0, 4);
        *ref_cyc = (float)(esp_cpu_get_cycle_count() - c0) / n;
        mmrt_use_ref = save;
        bad = 0;
        for (size_t i = 0; i < n; i++) bad += d0[i] != d1[i];
    }
    heap_caps_free(src);
    heap_caps_free(d0);
    heap_caps_free(d1);
    return bad;
}
