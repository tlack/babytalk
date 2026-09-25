# BabyTalk NIFs for AtomVM (ESP32-S3): speech to text on the board, callable from Erlang/Elixir.
# Registered through REGISTER_NIF_COLLECTION (a constructor), hence WHOLE_ARCHIVE.
idf_component_register(
    SRCS "babytalk_nifs.c"
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap"
    WHOLE_ARCHIVE
)
