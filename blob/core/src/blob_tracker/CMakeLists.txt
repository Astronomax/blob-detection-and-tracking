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

configure_target(blob_detector)

set_target_properties(blob_tracker PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(blob_tracker PUBLIC
        blob_detector
        "${CMAKE_THREAD_LIBS_INIT}"
        OpenMP::OpenMP_CXX
        Boost::headers
        Boost::system
        Boost::filesystem
)
