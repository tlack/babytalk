# BabyTalk NIFs for AtomVM (ESP32-S3): speech to text and text to speech on the board, callable
# from Erlang/Elixir. Engines: the shared stt_engine and sanotts components (repo-root
# components/). Audio: board_audio.h, implemented by board_audio.c for the Waveshare S3-CAM's
# chips, or by your own component (CONFIG_BABYTALK_BOARD_CUSTOM; see Kconfig).
# Registered through REGISTER_NIF_COLLECTION (a constructor), hence WHOLE_ARCHIVE.
set(srcs "babytalk_nifs.c")
if(NOT CONFIG_BABYTALK_BOARD_CUSTOM)
    list(APPEND srcs "board_audio.c")
endif()
idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap" "freertos" "log" "stt_engine" "sram_pool" "sanotts"
                  "esp_driver_i2c" "esp_driver_i2s" "nvs_flash"
    WHOLE_ARCHIVE
)
