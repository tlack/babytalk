// SRAM/PSRAM bandwidth across block sizes, two access patterns:
//   hot:    the same block over and over -- cache-resident once it fits in 64KB
//   stream: blocks walked across the whole buffer -- what weights streaming
//           through a layer actually see, since they never stay cached
#include <string.h>
#include "esp_heap_caps.h"
#include "bench.h"

#define TARGET_BYTES (32u * 1024 * 1024)  // stream this much per measurement

// Word-wise read sum. volatile sink keeps the loop from being optimized out.
static volatile uint32_t sink;

static void op_read(uint8_t *a, uint8_t *b, size_t n)
{
    (void)b;
    const uint32_t *p = (const uint32_t *)a;
    uint32_t s = 0;
    for (size_t i = 0; i < n / 4; i += 4) {
        s += p[i] + p[i + 1] + p[i + 2] + p[i + 3];
    }
    sink = s;
}

static void op_write(uint8_t *a, uint8_t *b, size_t n) { (void)b; memset(a, (int)n, n); }
static void op_copy(uint8_t *a, uint8_t *b, size_t n) { memcpy(b, a, n); }

typedef void (*op_fn)(uint8_t *, uint8_t *, size_t);

// Runs fn on n-byte blocks of a. span == n is the hot pattern; span > n walks
// the source across span bytes (span must be a multiple of n).
static void measure(const char *region, const char *op, op_fn fn, uint8_t *a, size_t span,
                    uint8_t *b, size_t n)
{
    unsigned reps = TARGET_BYTES / n;
    if (reps < 4) reps = 4;
    fn(a, b, n);  // warm
    int64_t t0 = bench_now_us();
    for (unsigned r = 0; r < reps; r++) fn(a + (size_t)r * n % span, b, n);
    int64_t us = bench_now_us() - t0;
    BENCH_RESULT("membw", "\"region\":\"%s\",\"op\":\"%s\",\"pattern\":\"%s\","
                 "\"bytes\":%u,\"mbps\":%.1f",
                 region, op, span == n ? "hot" : "stream", (unsigned)n,
                 bench_mbps(n * reps, us));
}

int bench_membw(int argc, char **argv)
{
    size_t max_kb = bench_arg_int(argc, argv, 1, 2048);
    // Internal SRAM can't hold 2 x max; cap its blocks at what fits.
    const size_t sram_max = 64 * 1024;

    uint8_t *sa = heap_caps_aligned_alloc(16, sram_max, MALLOC_CAP_INTERNAL);
    uint8_t *sb = heap_caps_aligned_alloc(16, sram_max, MALLOC_CAP_INTERNAL);
    uint8_t *pa = heap_caps_aligned_alloc(16, max_kb * 1024, MALLOC_CAP_SPIRAM);
    uint8_t *pb = heap_caps_aligned_alloc(16, max_kb * 1024, MALLOC_CAP_SPIRAM);
    if (!sa || !sb || !pa || !pb) {
        printf("alloc failed\n");
        heap_caps_free(sa); heap_caps_free(sb); heap_caps_free(pa); heap_caps_free(pb);
        return 1;
    }
    memset(sa, 1, sram_max); memset(pa, 1, max_kb * 1024);

    const size_t pspan = max_kb * 1024;
    for (size_t n = 4096; n <= sram_max; n *= 4) {
        measure("sram", "read", op_read, sa, n, sb, n);
        measure("sram", "write", op_write, sa, n, sb, n);
        measure("sram", "copy", op_copy, sa, n, sb, n);
    }
    for (size_t n = 16 * 1024; n <= pspan; n *= 4) {
        measure("psram", "read", op_read, pa, n, pb, n);
        measure("psram", "write", op_write, pa, n, pb, n);
        measure("psram", "copy", op_copy, pa, n, pb, n);
        if (n < pspan) {
            measure("psram", "read", op_read, pa, pspan, pb, n);
            measure("psram", "write", op_write, pa, pspan, pb, n);
        }
    }
    // The expert-tiling path: stream PSRAM weights into SRAM tiles.
    for (size_t n = 4096; n <= sram_max; n *= 4) {
        measure("psram->sram", "copy", op_copy, pa, pspan, sa, n);
    }

    heap_caps_free(sa); heap_caps_free(sb); heap_caps_free(pa); heap_caps_free(pb);
    return 0;
}
