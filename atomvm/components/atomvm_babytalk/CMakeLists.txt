# BabyTalk NIFs for AtomVM (ESP32-S3, ESP32-P4): speech to text and text to speech on the board,
# callable from Erlang/Elixir. Engines: the shared stt_engine and sanotts components (repo-root
# components/). Audio: board_audio.h, implemented by board_audio.c for the Waveshare S3-CAM's
# chips, board_audio_p4_wifi6.c for the Waveshare ESP32-P4-WIFI6's, or by your own component
# (CONFIG_BABYTALK_BOARD_CUSTOM; see Kconfig).
# Registered through REGISTER_NIF_COLLECTION (a constructor), hence WHOLE_ARCHIVE.
set(srcs "babytalk_nifs.c")
if(CONFIG_BABYTALK_BOARD_WAVESHARE_S3_CAM)
    list(APPEND srcs "board_audio.c")
elseif(CONFIG_BABYTALK_BOARD_WAVESHARE_P4_WIFI6)
    list(APPEND srcs "board_audio_p4_wifi6.c")
endif()
idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap" "freertos" "log" "stt_engine" "sram_pool" "sanotts"
                  "esp_driver_i2c" "esp_driver_i2s" "esp_driver_gpio" "nvs_flash"
    WHOLE_ARCHIVE
)
