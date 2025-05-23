# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex)

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
#link_directories (${APEX_BINARY_DIR}/src/apex_pthread_wrapper)

# Special handling for NVHPC, see below...
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Don't bother with GCC target offload
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(OMPT_TARGET_TEST ompt_target ompt_target_matmult ompt_target_daxpy ompt_target_vector_add)
endif()

set(example_programs
    ompt_thread
    ompt_parallel_region
    ompt_reduction
    ompt_task
    ompt_single
    ompt_sections
    ompt_sync_region_wait
    ompt_master
    ${OMPT_TARGET_TEST}
   )

foreach(example_program ${example_programs})
    set(sources ${example_program}.c)

    source_group("Source Files" FILES ${sources})

    add_executable(${example_program} ${sources})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${example_program} PRIVATE "-finstrument-functions")
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
        # Use the tool-enabled OpenMP runtime
        target_compile_options(${example_program} PRIVATE "-mp=ompt")
        target_link_options(${example_program} PRIVATE "-mp=ompt")
        target_link_libraries (${example_program} apex ${LIBS} ${APEX_STDCXX_LIB} m)
    else()
        target_link_libraries (${example_program} ${OMPT_LIBRARIES} apex ${LIBS} OpenMP::OpenMP_CXX ${APEX_STDCXX_LIB} m)
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
    if (BUILD_STATIC_EXECUTABLES)
        set_target_properties(${example_program} PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
    endif()
    add_dependencies (${example_program} apex)
    add_dependencies (tests ${example_program})
    add_test ("test_${example_program}" ${example_program})
    set_tests_properties("test_${example_program}" PROPERTIES TIMEOUT 30)
    set_tests_properties("test_${example_program}" PROPERTIES ENVIRONMENT "APEX_SCREEN_OUTPUT=1;APEX_OMPT_HIGH_OVERHEAD_EVENTS=1")
endforeach()

