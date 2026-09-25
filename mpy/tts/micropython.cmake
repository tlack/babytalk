# MicroPython user C module `tts` (bindings; the engine is the sanotts IDF component).
add_library(usermod_tts INTERFACE)
target_sources(usermod_tts INTERFACE ${CMAKE_CURRENT_LIST_DIR}/modtts.c)
target_include_directories(usermod_tts INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../components/sanotts)
target_link_libraries(usermod INTERFACE usermod_tts)
