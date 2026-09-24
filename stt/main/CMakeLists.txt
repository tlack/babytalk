idf_component_register(
    SRCS "main.cpp" "stt_core.c" "mic.c"
    INCLUDE_DIRS "."
    REQUIRES console esp_timer esp_partition spi_flash esp_psram heap esp-dl esp_wifi esp_netif esp_event nvs_flash lwip esp_driver_i2c esp_driver_i2s
)
