// Mapped-flash read bandwidth of the `model` partition: the candidate expert
// store that needs no SD card. Reads go flash -> MMU -> data cache; the region
// is larger than the cache, so this measures the flash bus.
#include "esp_partition.h"
#include "bench.h"

static volatile uint32_t sink;

int bench_flash(int argc, char **argv)
{
    size_t mb = bench_arg_int(argc, argv, 1, 4);
    const esp_partition_t *p =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!p) {
        printf("no model partition\n");
        return 1;
    }
    size_t len = mb * 1024 * 1024;
    if (len > p->size) len = p->size;

    const void *map;
    esp_partition_mmap_handle_t h;
    esp_err_t err = esp_partition_mmap(p, 0, len, ESP_PARTITION_MMAP_DATA, &map, &h);
    if (err != ESP_OK) {
        printf("mmap %u bytes failed: %s\n", (unsigned)len, esp_err_to_name(err));
        return 1;
    }

    const uint32_t *w = map;
    for (int pass = 0; pass < 2; pass++) {  // pass 0 includes cold-cache/MMU effects
        uint32_t s = 0;
        int64_t t0 = bench_now_us();
        for (size_t i = 0; i < len / 4; i += 4) {
            s += w[i] + w[i + 1] + w[i + 2] + w[i + 3];
        }
        int64_t us = bench_now_us() - t0;
        sink = s;
        BENCH_RESULT("flash", "\"op\":\"mmap_read\",\"pass\":%d,\"bytes\":%u,\"mbps\":%.1f",
                     pass, (unsigned)len, bench_mbps(len, us));
    }
    esp_partition_munmap(h);

    // esp_partition_read (SPI flash driver, no mapping) into SRAM, for comparison.
    static uint8_t buf[32 * 1024];
    int64_t t0 = bench_now_us();
    for (size_t off = 0; off < len; off += sizeof(buf)) {
        esp_partition_read(p, off, buf, sizeof(buf));
    }
    int64_t us = bench_now_us() - t0;
    BENCH_RESULT("flash", "\"op\":\"partition_read\",\"chunk\":%u,\"bytes\":%u,\"mbps\":%.1f",
                 (unsigned)sizeof(buf), (unsigned)len, bench_mbps(len, us));
    return 0;
}
