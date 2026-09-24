# All user C modules of the Sentry + stt firmware: the camera API (which pulls in
# mp_jpeg from next to it) and stt. CAMERA_API_DIR: the micropython-camera-API checkout.
if(DEFINED CAMERA_API_DIR AND EXISTS "${CAMERA_API_DIR}/micropython.cmake")
    include(${CAMERA_API_DIR}/micropython.cmake)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/stt/micropython.cmake)
# tts: only when build.sh found the sanoTTS sources (SANOTTS_DIR)
if(DEFINED SANOTTS_SRC AND EXISTS "${SANOTTS_SRC}/snt_nano.c")
    include(${CMAKE_CURRENT_LIST_DIR}/tts/micropython.cmake)
endif()
