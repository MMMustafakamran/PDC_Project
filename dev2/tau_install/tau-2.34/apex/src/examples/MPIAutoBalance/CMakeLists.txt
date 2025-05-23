# Make sure that spaces in linker lines don't cause CMake errors
#if (POLICY CMP0004)
#  cmake_policy(SET CMP0004 OLD)
#endif()

# Make sure the compiler can find include files from our Apex library.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
include_directories (. ${APEX_SOURCE_DIR}/src/apex ${APEX_SOURCE_DIR}/src/comm ${MPI_CXX_INCLUDE_PATH} ${OMPT_INCLUDE_DIR})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex)
link_directories (${APEX_BINARY_DIR}/src/wrappers)
link_directories (${APEX_BINARY_DIR}/src/comm/mpi)

# Add executable called "mpi_auto_balance" that is built from the source file
# "mpi_auto_balance.cpp". The extensions are automatically found.
add_executable (mpi_auto_balance mpi_auto_balance.cpp synchronous_policy.cpp)

# Link the executable to the Apex library.
# Note: that stupid -lapex_pthread_wrapper is required on the buildbot system.
target_link_libraries (mpi_auto_balance apex_pthread_wrapper apex_mpi apex_global_mpi apex ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${LIBS} -lapex_pthread_wrapper)
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(mpi_auto_balance PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()
add_dependencies (mpi_auto_balance apex apex_pthread_wrapper)
add_dependencies (examples mpi_auto_balance)

## Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

INSTALL(TARGETS mpi_auto_balance
  RUNTIME DESTINATION bin OPTIONAL
)
