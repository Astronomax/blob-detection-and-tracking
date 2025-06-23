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
add_test(blob_tracker_test blob_tracker_test)