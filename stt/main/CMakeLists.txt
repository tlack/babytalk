idf_component_register(
    SRCS "main.cpp" "stt_core.c"
    INCLUDE_DIRS "."
    REQUIRES console esp_timer esp_partition spi_flash esp_psram heap esp-dl esp_wifi esp_netif esp_event nvs_flash lwip
)
