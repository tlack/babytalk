# sanoTTS (github.com/Ampixa/sanoTTS, ESPHome component sources) + our wrapper.
# SANOTTS_SRC: a copy of esphome/components/sanotts prepared by mpy/build.sh; without
# it this component is empty and the firmware has no `tts` module. REQUIRES stays
# unconditional: IDF's early requirements pass does not see -D cache variables.
set(srcs "")
set(incs "")
if(DEFINED SANOTTS_SRC AND EXISTS "${SANOTTS_SRC}/snt_nano.c")
    set(S ${SANOTTS_SRC})
    set(srcs "tts_engine.c" "${S}/snt_nano.c" "${S}/snt_kernels_esp32s3.c" "${S}/snt_kernels_ref.c"
             "${S}/snt_matvec_esp32s3_asm.c" "${S}/snt_port_esphome.c" "${S}/sanotts_model_data.c"
             "${S}/nano_lex_g2p.c" "${S}/nano_lex_tables.c")
    set(incs "." "${S}")
endif()
idf_component_register(SRCS ${srcs} INCLUDE_DIRS ${incs} REQUIRES esp_timer heap esp_hw_support sram_pool)
if(srcs)
    # SNT_NANO_FAST_MATH: the only math config sanoTTS gated for this voice.
    # NANO_LEX_WITH_SILVER=0: gold dictionary only (1.5 MB instead of 3.05 MB of flash).
    target_compile_definitions(${COMPONENT_LIB} PRIVATE SNT_NANO_FAST_MATH NANO_LEX_WITH_SILVER=0)
    target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function -Wno-unused-variable)
endif()
