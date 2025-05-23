# Make sure that spaces in linker lines don't cause CMake errors
#if (POLICY CMP0004)
#  cmake_policy(SET CMP0004 OLD)
#endif()

# Make sure the compiler can find include files from our Apex library.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/comm ${MPI_CXX_INCLUDE_PATH})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/comm/mpi)

# Add executable called "mpi_global_test" that is built from the source file
# "mpi_test.cpp". The extensions are automatically found.
add_executable (mpi_global_test mpi_test.cpp)
add_dependencies (mpi_global_test apex apex_mpi apex_global_mpi)
add_dependencies (examples mpi_global_test)

# Link the executable to the Apex library.
target_link_libraries (mpi_global_test apex apex_mpi apex_global_mpi ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${LIBS} ${APEX_STDCXX_LIB} m)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(mpi_global_test PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

if(OpenMP_CXX_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND)

INSTALL(TARGETS mpi_global_test
  RUNTIME DESTINATION bin OPTIONAL
)
