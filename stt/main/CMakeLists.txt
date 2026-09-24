idf_component_register(
    SRCS "main.cpp" "stt_core.c" "mic.c" "kws.c" "../../mmrt/mmrt.c" "../../mmrt/mmrt_ref.c" "../../mmrt/mmrt_s3.c" "../../mmrt/mmrt_s3.S"
    INCLUDE_DIRS "." "../../mmrt"
    REQUIRES console esp_timer esp_partition spi_flash esp_psram heap esp_wifi esp_netif esp_event
             nvs_flash lwip esp_driver_i2c esp_driver_i2s esp-dsp
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MMRT_S3)
target_compile_definitions(${COMPONENT_LIB} PRIVATE STT_USE_DSP)
