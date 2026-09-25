# BabyTalk NIFs for AtomVM (ESP32-S3): speech to text and text to speech on the board, callable
# from Erlang/Elixir. Engines: the shared stt_engine and sanotts components (repo-root
# components/). Audio: our own drivers for the Waveshare S3-CAM's chips (board_audio.c).
# Registered through REGISTER_NIF_COLLECTION (a constructor), hence WHOLE_ARCHIVE.
idf_component_register(
    SRCS "babytalk_nifs.c" "board_audio.c"
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap" "freertos" "log" "stt_engine" "sram_pool" "sanotts"
                  "esp_driver_i2c" "esp_driver_i2s"
    WHOLE_ARCHIVE
)
