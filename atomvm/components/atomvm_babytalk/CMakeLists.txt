# BabyTalk NIFs for AtomVM (ESP32-S3, ESP32-P4): speech to text and text to speech on the board,
# callable from Erlang/Elixir. Engines: the shared stt_engine and sanotts components (repo-root
# components/). Audio: board_audio.h, implemented by board_audio.c (ES7210/ES8311 boards, pins
# and parts chosen at run time: board_audio_config.h) or by your own component
# (CONFIG_BABYTALK_BOARD_CUSTOM, which leaves board_audio.c empty; see Kconfig).
# Registered through REGISTER_NIF_COLLECTION (a constructor), hence WHOLE_ARCHIVE.
idf_component_register(
    SRCS "babytalk_nifs.c" "board_audio.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap" "freertos" "log" "stt_engine" "sram_pool" "sanotts"
                  "esp_driver_i2c" "esp_driver_i2s" "esp_driver_gpio" "nvs_flash"
    WHOLE_ARCHIVE
)
