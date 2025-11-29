set(COMMON_COMPILE_OPTIONS "")

set(RELEASE_COMPILE_OPTIONS -DBOOST_UBLAS_NDEBUG -DNDEBUG -O3 -ffast-math)

set(DEBUG_COMPILE_OPTIONS -O0 -g -ggdb -fno-omit-frame-pointer)

if(RASPBERRYPI)
    list(APPEND COMMON_COMPILE_OPTIONS ${RASPBERRY_COMMON_COMPILE_OPTIONS})
    list(APPEND RELEASE_COMPILE_OPTIONS ${RASPBERRY_RELEASE_COMPILE_OPTIONS})
endif()

function(configure_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PUBLIC ${COMMON_COMPILE_OPTIONS})
    target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:Release>:${RELEASE_COMPILE_OPTIONS}>)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:Debug>:${DEBUG_COMPILE_OPTIONS}>)
endfunction()

add_subdirectory(blob_detector)
add_subdirectory(blob_tracker)
