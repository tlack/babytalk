# BabyTalk NIFs for AtomVM (ESP32-S3): speech to text on the board, callable from Erlang/Elixir.
# The engine is the shared stt_engine component (repo-root components/); the microphone is
# the Waveshare S3-CAM's ES7210 (stt/main/mic.c). Registered through REGISTER_NIF_COLLECTION
# (a constructor), hence WHOLE_ARCHIVE.
get_filename_component(HERE "${CMAKE_CURRENT_LIST_DIR}" REALPATH)  # this dir may be a symlink
get_filename_component(R "${HERE}/../../.." ABSOLUTE)
idf_component_register(
    SRCS "babytalk_nifs.c" "${R}/stt/main/mic.c"
    INCLUDE_DIRS "${R}/stt/main"
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap" "freertos" "log" "stt_engine" "sram_pool"
                  "esp_driver_i2c" "esp_driver_i2s"
    WHOLE_ARCHIVE
)
