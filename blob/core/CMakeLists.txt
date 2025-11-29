cmake_minimum_required(VERSION 3.12)
project(blob_tracker)

cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

find_package(Boost 1.74.0 REQUIRED system filesystem)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest)

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/opt/boost")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/opt/boost/lib")

if(GTEST_FOUND)
    add_subdirectory(test)
else(GTEST_FOUND)
    message(WARNING "GTest not found")
endif(GTEST_FOUND)

add_subdirectory(src)
