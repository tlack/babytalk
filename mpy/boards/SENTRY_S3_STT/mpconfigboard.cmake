# Waveshare ESP32-S3-CAM (ESP32-S3R8: 8MB octal PSRAM, 16MB flash) with the stt module.
# = ESP32_GENERIC_S3 + SPIRAM_OCT (what the wt2 Sentry image is built as) + sdkconfig.board,
# minus Bluetooth: its controller reservation is internal RAM that the TTS arena needs.
set(IDF_TARGET esp32s3)

set(SDKCONFIG_DEFAULTS
    boards/sdkconfig.base
    boards/sdkconfig.spiram_sx
    boards/sdkconfig.240mhz
    boards/sdkconfig.spiram_oct
    ${MICROPY_BOARD_DIR}/sdkconfig.board
)

list(APPEND MICROPY_DEF_BOARD
    MICROPY_HW_BOARD_NAME="Waveshare ESP32-S3-CAM (stt)"
)
