# MMRT vector-kernel NIFs for AtomVM (ESP32-S3): int8/int4 matvec, matmul and friends on the
# S3's PIE SIMD unit, over tagged binaries. Registered through REGISTER_NIF_COLLECTION (a
# constructor), hence WHOLE_ARCHIVE.
idf_component_register(
    SRCS "mmrt_nifs.c"
    PRIV_REQUIRES "libatomvm" "avm_sys" "heap" "mmrt"
    WHOLE_ARCHIVE
)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
