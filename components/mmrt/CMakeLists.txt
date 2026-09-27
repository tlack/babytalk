# MMRT (mmrt/) as an ESP-IDF component, shared by the MicroPython firmware (mpy/) and the
# AtomVM firmware (atomvm/). On an ESP32-S3: MMRT_S3, the PIE kernels (exported, so the engines
# that set up their staging buffers see it), and MMRT_S3_HEAP_BUFFERS: staging buffers come
# from mmrt_s3_init() / a buffer provider (e.g. components/sram_pool) instead of static arrays,
# so firmwares that share internal RAM between engines can hand them over. On any other chip
# (the ESP32-P4, say): the portable kernels (mmrt_port.c: plain C, bit-exact with mmrt_ref.c),
# and on the P4 its 1x1 convolutions as row dot products on its vector unit (mmrt_p4.S).
get_filename_component(HERE "${CMAKE_CURRENT_LIST_DIR}" REALPATH)  # this dir may be a symlink
get_filename_component(R "${HERE}/../../mmrt" ABSOLUTE)
set(srcs "${R}/mmrt.c" "${R}/mmrt_ref.c")
if(IDF_TARGET STREQUAL "esp32s3")
    list(APPEND srcs "${R}/mmrt_s3.c" "${R}/mmrt_s3.S")
else()
    list(APPEND srcs "${R}/mmrt_port.c")
    if(IDF_TARGET STREQUAL "esp32p4")
        list(APPEND srcs "${R}/mmrt_p4.S")   # the dot product on the P4's vector unit
    endif()
endif()
idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "${R}"
    REQUIRES esp_timer heap freertos esp_hw_support
)
if(IDF_TARGET STREQUAL "esp32s3")
    target_compile_definitions(${COMPONENT_LIB} PUBLIC MMRT_S3 PRIVATE MMRT_S3_HEAP_BUFFERS)
endif()
if(IDF_TARGET STREQUAL "esp32p4")
    target_compile_definitions(${COMPONENT_LIB} PUBLIC MMRT_P4 PRIVATE MMRT_ROWDOT)
endif()
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
