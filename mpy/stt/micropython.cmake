# MicroPython user C module `stt` (bindings only; the engine is the stt_engine IDF component).
add_library(usermod_stt INTERFACE)
target_sources(usermod_stt INTERFACE ${CMAKE_CURRENT_LIST_DIR}/modstt.c)
target_include_directories(usermod_stt INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../components/stt_engine)
target_link_libraries(usermod INTERFACE usermod_stt)
