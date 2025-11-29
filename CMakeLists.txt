cmake_minimum_required(VERSION 3.12)
project(blob_tracker)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_INCLUDE_PATH})

include(CheckIncludeFile)
include(cmake/compiler.cmake)

configure_file(
    ${CMAKE_SOURCE_DIR}/blob/core/src/config.h.cmake
    ${CMAKE_BINARY_DIR}/config.h
)

include_directories(
    ${CMAKE_BINARY_DIR}
)

enable_testing()

add_subdirectory(blob)
