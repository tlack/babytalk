// Headroom with ESP-IDF loaded: what's left for a model after the OS.
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_image_format.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "sdkconfig.h"
#include "bench.h"

static void heap_report(const char *label, uint32_t caps)
{
    multi_heap_info_t info;
    heap_caps_get_info(&info, caps);
    BENCH_RESULT("mem", "\"heap\":\"%s\",\"total\":%u,\"free\":%u,\"largest\":%u,\"min_free\":%u",
                 label, (unsigned)heap_caps_get_total_size(caps), (unsigned)info.total_free_bytes,
                 (unsigned)info.largest_free_block, (unsigned)info.minimum_free_bytes);
}

int bench_mem(int argc, char **argv)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);
    BENCH_RESULT("mem", "\"idf\":\"%s\",\"cores\":%d,\"rev\":%d,\"cpu_mhz\":%d,"
                 "\"flash\":%u,\"psram\":%u,\"dcache\":%d,\"dcache_line\":%d,\"icache\":%d",
                 esp_get_idf_version(), chip.cores, chip.revision, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
                 (unsigned)flash_size, (unsigned)esp_psram_get_size(),
                 CONFIG_ESP32S3_DATA_CACHE_SIZE, CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE,
                 CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE);

    heap_report("internal", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    heap_report("dma", MALLOC_CAP_DMA);
    heap_report("psram", MALLOC_CAP_SPIRAM);

    // App image size vs its partition: the room left for the firmware to grow.
    const esp_partition_t *app = esp_ota_get_running_partition();
    esp_image_metadata_t meta = {0};
    const esp_partition_pos_t pos = {.offset = app->address, .size = app->size};
    uint32_t image_len = 0;
    if (esp_image_verify(ESP_IMAGE_VERIFY_SILENT, &pos, &meta) == ESP_OK) {
        image_len = meta.image_len;
    }
    BENCH_RESULT("mem", "\"app\":\"%s\",\"partition\":%u,\"image\":%u,\"app_free\":%u",
                 app->label, (unsigned)app->size, (unsigned)image_len,
                 (unsigned)(app->size - image_len));

    esp_partition_iterator_t it =
        esp_partition_find(ESP_PARTITION_TYPE_ANY, ESP_PARTITION_SUBTYPE_ANY, NULL);
    for (; it; it = esp_partition_next(it)) {
        const esp_partition_t *p = esp_partition_get(it);
        BENCH_RESULT("mem", "\"part\":\"%s\",\"type\":%d,\"subtype\":%d,\"offset\":%u,\"size\":%u",
                     p->label, p->type, p->subtype, (unsigned)p->address, (unsigned)p->size);
    }
    esp_partition_iterator_release(it);
    return 0;
}
