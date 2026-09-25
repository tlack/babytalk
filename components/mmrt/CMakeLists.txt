# MMRT (mmrt/) as an ESP-IDF component, shared by the MicroPython firmware (mpy/) and the
# AtomVM firmware (atomvm/). MMRT_S3: the ESP32-S3 PIE kernels. MMRT_S3_HEAP_BUFFERS: staging
# buffers come from mmrt_s3_init() / a buffer provider (e.g. components/sram_pool) instead of
# static arrays, so firmwares that share internal RAM between engines can hand them over.
get_filename_component(HERE "${CMAKE_CURRENT_LIST_DIR}" REALPATH)  # this dir may be a symlink
get_filename_component(R "${HERE}/../../mmrt" ABSOLUTE)
idf_component_register(
    SRCS "${R}/mmrt.c" "${R}/mmrt_ref.c" "${R}/mmrt_s3.c" "${R}/mmrt_s3.S"
    INCLUDE_DIRS "${R}"
    REQUIRES esp_timer heap freertos esp_hw_support
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MMRT_S3 MMRT_S3_HEAP_BUFFERS)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
