cmake_minimum_required(VERSION 3.12)
project(blob_tracker)

cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)

set(HEADERS
        hungarian/hungarian.hpp
        kalman_filter/kalman_filter.hpp
        blob_tracker.hpp
        )

set(SOURCES
        hungarian/hungarian.cpp
        kalman_filter/kalman_filter.cpp
        blob_tracker.cpp
        )

add_library(blob_tracker SHARED ${HEADERS} ${SOURCES})
target_compile_options(blob_tracker PRIVATE -fPIC)

if(RASPBERRYPI)
         target_compile_options(blob_tracker PUBLIC
        -march=armv8-a+crc+simd
        -mcpu=cortex-a72
        -mtune=cortex-a72
        -mfpu=neon
        -ftree-vectorize
        -mvectorize-with-neon-quad
        -ftree-vectorizer-verbose=1
        -funsafe-math-optimizations
        -mfloat-abi=hard
        -falign-functions=32
        -falign-loops=32
        -funroll-loops
        -DBOOST_UBLAS_NDEBUG
        -DNDEBUG
        -O3
        -ffast-math
        -g
        )
        target_compile_definitions(blob_tracker PUBLIC -DRASPBERRYPI)
else(RASPBERRYPI)
        target_compile_options(blob_tracker PUBLIC
        -O3
        -ffast-math
        -DBOOST_UBLAS_NDEBUG -DNDEBUG -g
        )
endif(RASPBERRYPI)
set_target_properties(blob_tracker PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(blob_tracker PUBLIC
        blob_detector
        "${CMAKE_THREAD_LIBS_INIT}"
        OpenMP::OpenMP_CXX
        Boost::headers
        Boost::system
        Boost::filesystem
        )
target_compile_definitions(blob_tracker PUBLIC -DPROJECT_ROOT="${PROJECT_ROOT}")