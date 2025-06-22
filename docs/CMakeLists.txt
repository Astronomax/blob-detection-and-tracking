find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API        
documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
         message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    else()
        set(DOXYGEN_IN ${PROJECT_ROOT}/docs/Doxyfile)
        #set(DOXYGEN_OUT ${PROJECT_ROOT}/docs/)
        #configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started.")

        add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
            WORKING_DIRECTORY ${PROJECT_ROOT}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
    endif()
endif()