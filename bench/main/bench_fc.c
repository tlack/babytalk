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
// Placement: weights in PSRAM (full layer), in mmapped flash (the `model`
// partition, read in place -- its contents are whatever is there, which doesn't
// matter for timing), and in internal SRAM (as many output rows as fit) for the
// ceiling.
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
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
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *wf = NULL;
    esp_partition_mmap_handle_t wf_handle = 0;
    if (!part || wbytes + PAD > part->size ||
        esp_partition_mmap(part, 0, wbytes + PAD, ESP_PARTITION_MMAP_DATA, &wf, &wf_handle) != ESP_OK) {
        printf("flash mmap failed; skipping flash placement\n");
        wf = NULL;
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
        fc_job_t jf = {wf, x, y, acc, in, out, frames};
        if (frames == 1) measure("ansi", run_ansi, "psram", &jp);
        measure("pie", run_pie, "psram", &jp);
        measure("pie_rows", run_pie_rows, "psram", &jp);
        if (wf) {
            measure("pie", run_pie, "flash", &jf);
            measure("pie_rows", run_pie_rows, "flash", &jf);
        }
        measure("pie", run_pie, "sram", &js);
        measure("pie_rows", run_pie_rows, "sram", &js);
    }
    rc = 0;
    if (wf) esp_partition_munmap(wf_handle);

out:
    heap_caps_free(wp); heap_caps_free(ws); heap_caps_free(x);
    heap_caps_free(y); heap_caps_free(acc);
    return rc;
}

// ---------------------------------------------------------------------------
// fc2: the same row-reuse kernel on both cores. Output rows are split in half,
// one task pinned per core, both released together; the wall time is the
// slower core's. Tells whether two cores double throughput or fight over the
// shared flash/PSRAM bus (MSPI + cache).
//
//   fc2 [in] [out] [frames]    defaults 1024 x 256, 64 frames

typedef struct {
    fc_job_t job;
    int reps;
    int64_t us;
    EventGroupHandle_t ev;
    int core;
} core_run_t;

#define EV_GO   (1 << 0)
#define EV_DONE(c) (1 << (1 + (c)))

static void core_task(void *arg)
{
    core_run_t *r = arg;
    xEventGroupWaitBits(r->ev, EV_GO, pdFALSE, pdTRUE, portMAX_DELAY);
    int64_t t0 = bench_now_us();
    for (int i = 0; i < r->reps; i++) run_pie_rows(&r->job);
    r->us = bench_now_us() - t0;
    xEventGroupSetBits(r->ev, EV_DONE(r->core));
    vTaskDelete(NULL);
}

// Runs two half-jobs concurrently, one per core; returns the wall time.
static int64_t run_dual(core_run_t runs[2], EventGroupHandle_t ev)
{
    xEventGroupClearBits(ev, EV_GO | EV_DONE(0) | EV_DONE(1));
    for (int c = 0; c < 2; c++) {
        runs[c].ev = ev;
        runs[c].core = c;
        xTaskCreatePinnedToCore(core_task, "fc2", 4096, &runs[c], 5, NULL, c);
    }
    vTaskDelay(pdMS_TO_TICKS(20));  // both parked on EV_GO
    int64_t t0 = bench_now_us();
    xEventGroupSetBits(ev, EV_GO);
    xEventGroupWaitBits(ev, EV_DONE(0) | EV_DONE(1), pdFALSE, pdTRUE, portMAX_DELAY);
    return bench_now_us() - t0;
}

static void measure_dual(const char *place, const int8_t *w0, const int8_t *w1, const int8_t *x,
                         int32_t *acc0, int32_t *acc1, int in, int out, int frames,
                         EventGroupHandle_t ev)
{
    int half = out / 2;
    fc_job_t full = {w0, x, NULL, acc0, in, half * 2, frames};
    // One-core reference: the whole layer on this core, same weights.
    int64_t t0 = bench_now_us();
    run_pie_rows(&full);
    int64_t once = bench_now_us() - t0;
    int reps = once > 0 ? MIN_US / once : 100;
    if (reps < 1) reps = 1;
    t0 = bench_now_us();
    for (int r = 0; r < reps; r++) run_pie_rows(&full);
    int64_t us1 = bench_now_us() - t0;

    core_run_t runs[2] = {
        {.job = {w0, x, NULL, acc0, in, half, frames}, .reps = reps},
        {.job = {w1 + (size_t)half * in, x, NULL, acc1, in, half, frames}, .reps = reps},
    };
    int64_t us2 = run_dual(runs, ev);

    double macs = (double)in * half * 2 * frames * reps;
    BENCH_RESULT("fc2", "\"weights\":\"%s\",\"in\":%d,\"out\":%d,\"frames\":%d,"
                 "\"gmacs_1core\":%.3f,\"gmacs_2core\":%.3f,\"speedup\":%.2f,"
                 "\"core0_us\":%lld,\"core1_us\":%lld",
                 place, in, half * 2, frames, macs / us1 / 1000.0, macs / us2 / 1000.0,
                 (double)us1 / us2, (long long)runs[0].us, (long long)runs[1].us);
}

int bench_fc2(int argc, char **argv)
{
    int in = bench_arg_int(argc, argv, 1, 1024) & ~15;
    int out = bench_arg_int(argc, argv, 2, 256) & ~1;
    int max_frames = bench_arg_int(argc, argv, 3, 64);
    if (in < 16 || out < 2 || max_frames < 1) return 1;
    int sram_out = (SRAM_BYTES / in < out ? SRAM_BYTES / in : out) & ~1;

    size_t wbytes = (size_t)in * out;
    size_t acc_bytes = sizeof(int32_t) * (out / 2) * max_frames;
    int8_t *wp = heap_caps_aligned_alloc(16, wbytes + PAD, MALLOC_CAP_SPIRAM);
    int8_t *ws = heap_caps_aligned_alloc(16, (size_t)in * sram_out + PAD, MALLOC_CAP_INTERNAL);
    int8_t *x = heap_caps_aligned_alloc(16, (size_t)in * max_frames + PAD, MALLOC_CAP_INTERNAL);
    // acc0 also serves the one-core reference, which needs the full width.
    int32_t *acc0 = heap_caps_malloc(acc_bytes * 2, MALLOC_CAP_INTERNAL);
    int32_t *acc1 = heap_caps_malloc(acc_bytes, MALLOC_CAP_INTERNAL);
    EventGroupHandle_t ev = xEventGroupCreate();
    const void *wf = NULL;
    esp_partition_mmap_handle_t wf_handle = 0;
    int rc = 1;
    if (!wp || !ws || !x || !acc0 || !acc1 || !ev) {
        printf("alloc failed\n");
        goto out;
    }
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part || wbytes + PAD > part->size ||
        esp_partition_mmap(part, 0, wbytes + PAD, ESP_PARTITION_MMAP_DATA, &wf, &wf_handle) != ESP_OK) {
        printf("flash mmap failed; skipping flash placements\n");
        wf = NULL;
    }
    fill(wp, wbytes, 1);
    fill(ws, (size_t)in * sram_out, 2);
    fill(x, (size_t)in * max_frames, 3);

    const int frame_counts[] = {1, 16, 64};
    for (size_t i = 0; i < sizeof(frame_counts) / sizeof(frame_counts[0]); i++) {
        int f = frame_counts[i];
        if (f > max_frames) break;
        measure_dual("sram", ws, ws, x, acc0, acc1, in, sram_out, f, ev);
        measure_dual("psram", wp, wp, x, acc0, acc1, in, out, f, ev);
        if (wf) {
            measure_dual("flash", wf, wf, x, acc0, acc1, in, out, f, ev);
            // core 0 streams flash while core 1 streams PSRAM: same bus.
            measure_dual("flash+psram", wf, wp, x, acc0, acc1, in, out, f, ev);
        }
    }
    rc = 0;

out:
    if (wf) esp_partition_munmap(wf_handle);
    if (ev) vEventGroupDelete(ev);
    heap_caps_free(wp); heap_caps_free(ws); heap_caps_free(x);
    heap_caps_free(acc0); heap_caps_free(acc1);
    return rc;
}
