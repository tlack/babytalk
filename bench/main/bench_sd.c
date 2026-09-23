// SD read bandwidth on the Waveshare S3 cam's 1-bit SDMMC (CLK=16 CMD=43 D0=44).
//   raw: sdmmc_read_sectors from the middle of the card (read-only, FS untouched)
//   fat: a test file written once to /sdcard/mmbench.bin, then read back
// Each at several chunk sizes, into internal DMA SRAM vs PSRAM (a PSRAM
// destination may bounce through internal RAM inside the driver -- measuring
// that cost is the point).
//
// Never formats: if the card doesn't mount as FAT, the bench fails.
#include <string.h>
#include <sys/stat.h>
#include "driver/sdmmc_host.h"
#include "esp_heap_caps.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "bench.h"

#define MOUNT      "/sdcard"
#define TEST_FILE  MOUNT "/mmbench.bin"
#define RAW_BYTES  (4u * 1024 * 1024)
#define MAX_CHUNK  (64u * 1024)

static const size_t CHUNKS[] = {4096, 16384, 65536};

static void raw_read(sdmmc_card_t *card, const char *dst, uint8_t *buf, size_t chunk)
{
    size_t secs = chunk / card->csd.sector_size;
    size_t start = card->csd.capacity / 2;  // middle of the card, clear of FAT/root dir
    int64_t t0 = bench_now_us();
    for (size_t done = 0; done < RAW_BYTES; done += chunk, start += secs) {
        if (sdmmc_read_sectors(card, buf, start, secs) != ESP_OK) {
            printf("raw read failed at sector %u\n", (unsigned)start);
            return;
        }
    }
    int64_t us = bench_now_us() - t0;
    BENCH_RESULT("sd", "\"op\":\"raw_read\",\"dst\":\"%s\",\"chunk\":%u,\"bytes\":%u,\"mbps\":%.2f",
                 dst, (unsigned)chunk, RAW_BYTES, bench_mbps(RAW_BYTES, us));
}

static int ensure_test_file(size_t bytes, uint8_t *buf)
{
    struct stat st;
    if (stat(TEST_FILE, &st) == 0 && (size_t)st.st_size == bytes) return 0;

    FILE *f = fopen(TEST_FILE, "wb");
    if (!f) return 1;
    setvbuf(f, NULL, _IONBF, 0);
    for (size_t i = 0; i < MAX_CHUNK; i++) buf[i] = (uint8_t)i;
    int64_t t0 = bench_now_us();
    for (size_t done = 0; done < bytes; done += MAX_CHUNK) {
        if (fwrite(buf, 1, MAX_CHUNK, f) != MAX_CHUNK) {
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    int64_t us = bench_now_us() - t0;
    BENCH_RESULT("sd", "\"op\":\"fat_write\",\"chunk\":%u,\"bytes\":%u,\"mbps\":%.2f",
                 MAX_CHUNK, (unsigned)bytes, bench_mbps(bytes, us));
    return 0;
}

static void fat_read(const char *dst, uint8_t *buf, size_t chunk, size_t bytes)
{
    FILE *f = fopen(TEST_FILE, "rb");
    if (!f) return;
    setvbuf(f, NULL, _IONBF, 0);  // no stdio copy: straight into buf
    int64_t t0 = bench_now_us();
    size_t total = 0, n;
    while ((n = fread(buf, 1, chunk, f)) > 0) total += n;
    int64_t us = bench_now_us() - t0;
    fclose(f);
    BENCH_RESULT("sd", "\"op\":\"fat_read\",\"dst\":\"%s\",\"chunk\":%u,\"bytes\":%u,\"mbps\":%.2f",
                 dst, (unsigned)chunk, (unsigned)total, bench_mbps(total, us));
}

int bench_sd(int argc, char **argv)
{
    int freq_khz = bench_arg_int(argc, argv, 1, SDMMC_FREQ_HIGHSPEED);
    size_t file_bytes = (size_t)bench_arg_int(argc, argv, 2, 8) * 1024 * 1024;

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = freq_khz;
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = GPIO_NUM_16;
    slot.cmd = GPIO_NUM_43;
    slot.d0 = GPIO_NUM_44;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    const esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 2,
        .allocation_unit_size = 16 * 1024,
    };

    sdmmc_card_t *card;
    esp_err_t err = esp_vfs_fat_sdmmc_mount(MOUNT, &host, &slot, &mount_cfg, &card);
    if (err != ESP_OK) {
        printf("mount failed: %s\n", esp_err_to_name(err));
        return 1;
    }
    BENCH_RESULT("sd", "\"card\":\"%s\",\"capacity_mb\":%u,\"req_khz\":%d,\"real_khz\":%d,\"width\":1",
                 card->cid.name,
                 (unsigned)((uint64_t)card->csd.capacity * card->csd.sector_size >> 20),
                 freq_khz, card->real_freq_khz);

    int rc = 1;
    uint8_t *sram = heap_caps_aligned_alloc(64, MAX_CHUNK, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    uint8_t *psram = heap_caps_aligned_alloc(64, MAX_CHUNK, MALLOC_CAP_SPIRAM);
    if (!sram || !psram) {
        printf("alloc failed\n");
        goto out;
    }

    for (size_t i = 0; i < sizeof(CHUNKS) / sizeof(CHUNKS[0]); i++) {
        raw_read(card, "sram", sram, CHUNKS[i]);
        raw_read(card, "psram", psram, CHUNKS[i]);
    }
    if (ensure_test_file(file_bytes, sram) != 0) {
        printf("test file write failed\n");
        goto out;
    }
    for (size_t i = 0; i < sizeof(CHUNKS) / sizeof(CHUNKS[0]); i++) {
        fat_read("sram", sram, CHUNKS[i], file_bytes);
        fat_read("psram", psram, CHUNKS[i], file_bytes);
    }
    rc = 0;

out:
    heap_caps_free(sram);
    heap_caps_free(psram);
    esp_vfs_fat_sdcard_unmount(MOUNT, card);
    return rc;
}
