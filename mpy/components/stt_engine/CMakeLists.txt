# mmrt runtime + Citrinet front end + keyword spotting + the engine, as one IDF
# component (outside MicroPython's usermod so its qstr scan never sees the tables/asm).
set(R ${CMAKE_CURRENT_LIST_DIR}/../../..)
idf_component_register(
    SRCS "stt_engine.c"
         "${R}/mmrt/mmrt.c" "${R}/mmrt/mmrt_ref.c" "${R}/mmrt/mmrt_s3.c" "${R}/mmrt/mmrt_s3.S"
         "${R}/stt/main/stt_core.c" "${R}/stt/main/kws.c"
    INCLUDE_DIRS "." "${R}/mmrt" "${R}/stt/main"
    REQUIRES esp_partition esp_timer heap freertos spi_flash esp_psram esp_hw_support sram_pool
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MMRT_S3 MMRT_S3_HEAP_BUFFERS STT_USE_DSP)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
