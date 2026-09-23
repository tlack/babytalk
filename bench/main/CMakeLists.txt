idf_component_register(
    SRCS "main.c" "bench_mem.c" "bench_membw.c" "bench_flash.c" "bench_sd.c" "bench_fc.c"
    INCLUDE_DIRS "."
    REQUIRES console esp_timer esp_psram esp_partition spi_flash app_update bootloader_support
             fatfs sdmmc esp_driver_sdmmc vfs heap
)
