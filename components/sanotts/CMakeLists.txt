# sanoTTS (github.com/Ampixa/sanoTTS, ESPHome component sources) + our wrapper, shared by
# the MicroPython (mpy/) and AtomVM (atomvm/) firmwares. SANOTTS_SRC: a copy of
# esphome/components/sanotts made by prepare.sh; without it, a stub (tts_stub.c: every say
# fails with -4) -- a speech-to-text-only firmware.
# SANOTTS_BSS_PSRAM: put the runtime's ~104 KB of static buffers in PSRAM (linker.lf; needs
# CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY) -- the synthesis arena stays in internal SRAM.
# REQUIRES stays unconditional: IDF's early requirements pass does not see -D cache variables.
set(srcs "tts_stub.c")
set(incs ".")
set(tts 0)
if(DEFINED SANOTTS_SRC AND EXISTS "${SANOTTS_SRC}/snt_nano.c")
    set(S ${SANOTTS_SRC})
    set(srcs "tts_engine.c" "${S}/snt_nano.c" "${S}/snt_kernels_esp32s3.c" "${S}/snt_kernels_ref.c"
             "${S}/snt_matvec_esp32s3_asm.c" "${S}/snt_port_esphome.c" "${S}/sanotts_model_data.c"
             "${S}/nano_lex_g2p.c" "${S}/nano_lex_tables.c")
    set(incs "." "${S}")
    set(tts 1)
    if(IDF_TARGET STREQUAL "esp32p4")   # BabyTalk's P4 int8 kernels in place of the scalar ones
        list(APPEND srcs "snt_kernels_esp32p4.c" "snt_matvec_esp32p4.S")
    endif()
    # prepare.sh ... heart / heart4: a larger voice (${S}/voice/) instead of the component's own,
    # its weights in the "voice" flash partition (voice.bin), not the app; the frame-batched
    # stages (snt_nano.patch) and, on the P4, both cores (snt_port_p4.c)
    if(EXISTS "${S}/voice/voice.bin")
        set(voice_part 1)
        list(REMOVE_ITEM srcs "${S}/sanotts_model_data.c")
        list(APPEND srcs "snt_q4.c")
        if(IDF_TARGET STREQUAL "esp32p4")
            list(APPEND srcs "snt_q4_esp32p4.S" "snt_port_p4.c")
            list(REMOVE_ITEM srcs "${S}/snt_port_esphome.c")
        endif()
    endif()
endif()
set(frags "")
if(tts AND SANOTTS_BSS_PSRAM)
    set(frags "linker.lf")
endif()
idf_component_register(SRCS ${srcs} INCLUDE_DIRS ${incs} LDFRAGMENTS ${frags}
                       REQUIRES esp_timer heap esp_hw_support esp_partition sram_pool)
if(tts)
    # SNT_NANO_FAST_MATH: the only math config sanoTTS gated for this voice.
    # NANO_LEX_WITH_SILVER=0: gold dictionary only (1.5 MB instead of 3.05 MB of flash).
    target_compile_definitions(${COMPONENT_LIB} PRIVATE SNT_NANO_FAST_MATH NANO_LEX_WITH_SILVER=0)
    if(IDF_TARGET STREQUAL "esp32p4")
        target_compile_definitions(${COMPONENT_LIB} PRIVATE SANOTTS_P4_SIMD=1)
    endif()
    if(voice_part)
        # the runtime's weight format from the voice's header (2: BabyTalk's 4-bit, 0: int8)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE SANOTTS_VOICE_PARTITION=1 SNT_NANO_PW_BATCH=8)
        file(STRINGS "${S}/nano_q8_meta.h" wf REGEX "#define NANO_WEIGHT_FORMAT 2")
        if(wf)
            target_compile_definitions(${COMPONENT_LIB} PRIVATE SNT_NANO_W_Q4)
        endif()
    endif()
    target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function -Wno-unused-variable)
endif()
