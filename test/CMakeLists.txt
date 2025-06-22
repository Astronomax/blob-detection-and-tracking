project(blob_tracker_test)
add_executable(blob_tracker_test test.cpp
        ../blob_detector/blob_detector.hpp
        ../blob_tracker/blob_tracker.hpp
        ../blob_tracker/blob_tracker.cpp
        ../blob_tracker/hungarian/hungarian.hpp
        ../blob_tracker/hungarian/hungarian.cpp
        ../blob_tracker/kalman_filter/kalman_filter.hpp
        ../blob_tracker/kalman_filter/kalman_filter.cpp)
target_link_libraries(blob_tracker_test
        ${GTEST_LIBRARIES}
        Boost::headers
        Boost::system
        Boost::filesystem
        )
add_test(blob_tracker_test blob_tracker_test)