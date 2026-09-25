# Speech to text engine: Citrinet front end + keyword spotting + the engine (map the model
# partition, PCM -> text, wake-phrase score, background task). Shared by the MicroPython
# (mpy/) and AtomVM (atomvm/) firmwares; MMRT itself is the mmrt component.
set(R ${CMAKE_CURRENT_LIST_DIR}/../..)
idf_component_register(
    SRCS "stt_engine.c" "${R}/stt/main/stt_core.c" "${R}/stt/main/kws.c"
    INCLUDE_DIRS "." "${R}/stt/main"
    REQUIRES mmrt sram_pool esp_partition esp_timer heap freertos spi_flash esp_psram esp_hw_support
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE STT_USE_DSP)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
