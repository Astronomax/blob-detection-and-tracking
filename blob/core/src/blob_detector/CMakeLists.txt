cmake_minimum_required(VERSION 3.12)
project(blob_detector)

cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)

set(HEADERS
        ../blob/blob.hpp
        blob_detector.hpp
        fft/fft_fast.hpp
        fft/fft_slow.hpp
        kdtree/kdtree.h
)

add_library(blob_detector SHARED ${HEADERS})
target_compile_options(blob_detector PRIVATE -fPIC)

configure_target(blob_detector)

set_target_properties(blob_detector PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(blob_detector PUBLIC
        "${CMAKE_THREAD_LIBS_INIT}"
        OpenMP::OpenMP_CXX
        Boost::headers
        Boost::system
        Boost::filesystem
)
