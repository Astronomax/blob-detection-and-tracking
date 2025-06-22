cmake_minimum_required(VERSION 3.0)
project(blob_tracker)

cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)

set(HEADERS
        ../blob/blob.hpp
        blob_detector.hpp
        fft/fft_fast.hpp
        fft/fft_slow.hpp
        kdtree/kdtree.hpp
        )

add_library(blob_detector ${HEADERS})
if(RASPBERRYPI)
         target_compile_options(blob_detector PUBLIC
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
        target_compile_definitions(blob_detector PUBLIC -DRASPBERRYPI)
else(RASPBERRYPI)
        target_compile_options(blob_detector PUBLIC
        -O3
        -ffast-math
        -DBOOST_UBLAS_NDEBUG -DNDEBUG -g
        )
endif(RASPBERRYPI)
set_target_properties(blob_detector PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(blob_detector PUBLIC
        "${SDL2_INCLUDE_DIRS}"
        "${GLEW_INCLUDE_DIRS}"
        "${OPENGL_INCLUDE_DIRS}"
        )
target_link_libraries(blob_detector PUBLIC
        "${GLEW_LIBRARIES}"
        "${SDL2_LIBRARIES}"
        "${OPENGL_LIBRARIES}"
        "${CMAKE_THREAD_LIBS_INIT}"
        OpenMP::OpenMP_CXX
        Boost::headers
        Boost::system
        Boost::filesystem
        )
target_compile_definitions(blob_detector PUBLIC -DPROJECT_ROOT="${PROJECT_ROOT}")