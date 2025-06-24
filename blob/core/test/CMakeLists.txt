add_executable(blob_tracker_test test.cpp
        ../src/blob_detector/blob_detector.hpp
        ../src/blob_tracker/blob_tracker.hpp
        ../src/blob_tracker/blob_tracker.cpp
        ../src/blob_tracker/hungarian/hungarian.hpp
        ../src/blob_tracker/hungarian/hungarian.cpp
        ../src/blob_tracker/kalman_filter/kalman_filter.hpp
        ../src/blob_tracker/kalman_filter/kalman_filter.cpp)

target_link_libraries(blob_tracker_test
        ${GTEST_LIBRARIES}
        Boost::headers
        Boost::system
        Boost::filesystem
        )

add_test(NAME blob_tracker_test COMMAND blob_tracker_test)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.10)
    include(GoogleTest)
    gtest_discover_tests(blob_tracker_test
        PROPERTIES
        DISCOVERY_TIMEOUT 60
    )
endif()
