# Make sure that spaces in linker lines don't cause CMake errors
#if (POLICY CMP0004)
#  cmake_policy(SET CMP0004 OLD)
#endif()

# Make sure the compiler can find include files from our Apex library.
include_directories (${APEX_SOURCE_DIR}/src/apex ${OMPT_INCLUDE_DIR} ${APEX_SOURCE_DIR}/src/comm ${MPI_CXX_INCLUDE_PATH})

# Make sure the linker can find the Apex library once it is built.
link_directories (${APEX_BINARY_DIR}/src/apex ${APEX_BINARY_DIR}/src/comm/mpi)

# Add executable called "lulesh_MPI_OpenMP_2.0" that is built from the source file
# "lulesh.cc". The extensions are automatically found.
add_executable (lulesh_MPI_OpenMP_2.0 lulesh.cc lulesh-comm.cc lulesh-init.cc lulesh-util.cc lulesh-viz.cc  synchronous_policy.cpp)
add_dependencies (lulesh_MPI_OpenMP_2.0 apex)
add_dependencies (examples lulesh_MPI_OpenMP_2.0)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3 -DAPEX_ENABLE_MPI=1 -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -DAPEX_ENABLE_MPI=1 -Wall")

# Link the executable to the Apex library.
target_link_libraries (lulesh_MPI_OpenMP_2.0 apex apex_mpi apex_global_mpi ${OMPT_LIBRARIES}
${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${LIBS})
if (BUILD_STATIC_EXECUTABLES)
    set_target_properties(lulesh_MPI_OpenMP_2.0 PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
endif()

INSTALL(TARGETS lulesh_MPI_OpenMP_2.0
  RUNTIME DESTINATION bin OPTIONAL
)
