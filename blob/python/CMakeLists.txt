find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

set(MODULE_NAME "_blob_internal")

pybind11_add_module(${MODULE_NAME}
    wrapper.cpp
)

target_compile_options(${MODULE_NAME} PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

set_target_properties(${MODULE_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
    OUTPUT_NAME ${MODULE_NAME}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_module"
    PREFIX ""
    SUFFIX ".so"
)

target_link_libraries(${MODULE_NAME} PRIVATE
    blob_detector
    blob_tracker
    pybind11::module
)

target_include_directories(${MODULE_NAME} PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${MODULE_NAME} PRIVATE stdc++fs)
endif()

if(WIN32)
    target_compile_definitions(${MODULE_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

add_custom_target(test_module
    COMMAND ${Python_EXECUTABLE} -c 
        "import sys; sys.path.append('${CMAKE_BINARY_DIR}/python_module'); import ${MODULE_NAME}; print('Module loaded successfully!')"
    DEPENDS ${MODULE_NAME}
    COMMENT "Testing Python module import"
    VERBATIM
)

set(TARGET_PKG_DIR "${CMAKE_SOURCE_DIR}/blob/_internal")

add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TARGET_PKG_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:${MODULE_NAME}>"
        "${TARGET_PKG_DIR}/_blob_internal$<TARGET_FILE_SUFFIX:${MODULE_NAME}>"
    COMMENT "Copying binary to package directory"
)

install(TARGETS ${MODULE_NAME}
    LIBRARY DESTINATION "${Python_SITEARCH}/blob_detector"
)
