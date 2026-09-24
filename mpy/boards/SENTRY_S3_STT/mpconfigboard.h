#ifndef MICROPY_HW_BOARD_NAME
#define MICROPY_HW_BOARD_NAME               "Waveshare ESP32-S3-CAM (stt)"
#endif
#define MICROPY_HW_MCU_NAME                 "ESP32S3"

#define MICROPY_HW_ENABLE_UART_REPL         (1)

// REPL on the chip's USB-Serial-JTAG (303a:1001, the same USB identity as ROM download
// mode) instead of TinyUSB CDC (303a:4001): esptool can reset it with no BOOT button,
// boot/crash logs stay visible, and under WSL/usbipd there is only one device to bind.
#define MICROPY_HW_ENABLE_USBDEV            (0)

#define MICROPY_HW_I2C0_SCL                 (9)
#define MICROPY_HW_I2C0_SDA                 (8)
