// int8 fully-connected MAC throughput -- the budget that sets active params.
//
//   fc [in] [out] [frames]     defaults 1024 x 1024 (1MB of weights), 16 frames
//
// Kernels:
//   ansi      ESP-NN plain C reference, called once per frame
//   pie       ESP-NN PIE (SIMD) FC, called once per frame: the whole weight
//             matrix streams through the cache once per frame
//   pie_rows  PIE dot product, row-major: each weight row is fetched once and
//             applied to every frame in the chunk (weight reuse, PLAN §2 wall 2)
// Placement: weights in PSRAM (full layer), and in internal SRAM (as many
// output rows as fit) for the no-PSRAM ceiling.
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_nn.h"
#include "esp_nn_ansi_headers.h"
#include "bench.h"

// ESP-NN internal kernel (global in its .S, not in a public header). `a` must
// be 16-byte aligned; both operands are over-read by up to 31 bytes.
extern int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a, const int8_t *b,
                                               int32_t len_div16);

#define PAD        32
#define SRAM_BYTES (64 * 1024)
#define MIN_US     300000  // run each config at least this long

typedef struct {
    const int8_t *w;
    const int8_t *x;
    int8_t *y;
    int32_t *acc;
    int in, out, frames;
} fc_job_t;

static void run_ansi(const fc_job_t *j)
{
    for (int f = 0; f < j->frames; f++) {
        esp_nn_fully_connected_s8_ansi(j->x + f * j->in, 0, j->in, j->w, 0, NULL,
                                       j->y + f * j->out, j->out, 0, -8, 1 << 30, -128, 127);
    }
}

static void run_pie(const fc_job_t *j)
{
    for (int f = 0; f < j->frames; f++) {
        esp_nn_fully_connected_s8(j->x + f * j->in, 0, j->in, j->w, 0, NULL,
                                  j->y + f * j->out, j->out, 0, -8, 1 << 30, -128, 127);
    }
}

static void run_pie_rows(const fc_job_t *j)
{
    for (int ch = 0; ch < j->out; ch++) {
        const int8_t *row = j->w + ch * j->in;
        for (int f = 0; f < j->frames; f++) {
            j->acc[f * j->out + ch] = esp_nn_dot_s8_unaligned_esp32s3(j->x + f * j->in, row, j->in >> 4);
        }
    }
}

typedef void (*kern_fn)(const fc_job_t *);

static void measure(const char *kern, kern_fn fn, const char *place, fc_job_t *j)
{
    int64_t t0 = bench_now_us();
    fn(j);
    int64_t once = bench_now_us() - t0;
    int reps = once > 0 ? MIN_US / once : 100;
    if (reps < 1) reps = 1;

    t0 = bench_now_us();
    for (int r = 0; r < reps; r++) fn(j);
    int64_t us = bench_now_us() - t0;

    double macs = (double)j->in * j->out * j->frames * reps;
    double wbytes = (double)j->in * j->out * reps;  // weight bytes per full pass
    BENCH_RESULT("fc", "\"kernel\":\"%s\",\"weights\":\"%s\",\"in\":%d,\"out\":%d,\"frames\":%d,"
                 "\"us_per_call\":%.1f,\"gmacs\":%.3f,\"weight_mbps\":%.1f",
                 kern, place, j->in, j->out, j->frames, (double)us / reps,
                 macs / us / 1000.0, bench_mbps(wbytes, us));
}

static void fill(int8_t *p, size_t n, uint32_t seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = (int8_t)(seed >> 24);
    }
}

int bench_fc(int argc, char **argv)
{
    int in = bench_arg_int(argc, argv, 1, 1024) & ~15;  // PIE path needs in % 16 == 0
    int out = bench_arg_int(argc, argv, 2, 1024);
    int max_frames = bench_arg_int(argc, argv, 3, 16);
    if (in < 16 || out < 1 || max_frames < 1) return 1;
    int sram_out = SRAM_BYTES / in < out ? SRAM_BYTES / in : out;

    size_t wbytes = (size_t)in * out;
    int8_t *wp = heap_caps_aligned_alloc(16, wbytes + PAD, MALLOC_CAP_SPIRAM);
    int8_t *ws = heap_caps_aligned_alloc(16, (size_t)in * sram_out + PAD, MALLOC_CAP_INTERNAL);
    int8_t *x = heap_caps_aligned_alloc(16, (size_t)in * max_frames + PAD, MALLOC_CAP_INTERNAL);
    int8_t *y = heap_caps_malloc((size_t)out * max_frames, MALLOC_CAP_INTERNAL);
    int32_t *acc = heap_caps_malloc(sizeof(int32_t) * out * max_frames, MALLOC_CAP_INTERNAL);
    int rc = 1;
    if (!wp || !ws || !x || !y || !acc) {
        printf("alloc failed (weights %u bytes)\n", (unsigned)wbytes);
        goto out;
    }
    fill(wp, wbytes, 1);
    fill(ws, (size_t)in * sram_out, 2);
    fill(x, (size_t)in * max_frames, 3);

    const int frame_counts[] = {1, 4, 16, 64};
    for (size_t i = 0; i < sizeof(frame_counts) / sizeof(frame_counts[0]); i++) {
        int frames = frame_counts[i];
        if (frames > max_frames) break;
        fc_job_t jp = {wp, x, y, acc, in, out, frames};
        fc_job_t js = {ws, x, y, acc, in, sram_out, frames};
        if (frames == 1) measure("ansi", run_ansi, "psram", &jp);
        measure("pie", run_pie, "psram", &jp);
        measure("pie_rows", run_pie_rows, "psram", &jp);
        measure("pie", run_pie, "sram", &js);
        measure("pie_rows", run_pie_rows, "sram", &js);
    }
    rc = 0;

out:
    heap_caps_free(wp); heap_caps_free(ws); heap_caps_free(x);
    heap_caps_free(y); heap_caps_free(acc);
    return rc;
}
