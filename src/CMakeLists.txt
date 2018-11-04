# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project(TensorFlowExperiments)

if(MSVC)
add_definitions(-D_USE_MATH_DEFINES)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest /W0")
add_definitions(-DMKL_ILP64)
add_definitions(-DMKL_DIRECT_CALL)
endif(MSVC)

add_subdirectory(Geo)

set(BUILD_TESTING OFF CACHE BOOL "Use some expat option")
set(BUILD_BENCHMARKS OFF CACHE BOOL "Use some expat option 2")
set(MINIGLOG on)

add_definitions(-DPROTOBUF_USE_DLLS)
add_definitions(-DEIGEN_USE_MKL_ALL)
add_definitions(-DEIGEN_USE_LAPACKE)
add_definitions(-DMKL_ILP64)
add_definitions(-DMKL_DIRECT_CALL)
add_definitions(-DNOMINMAX -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=std::int64_t)
include_directories (
  $ENV{MKL_ROOT_DIR}/mkl/include
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/tensorflow/include)

set (MKL_LIB_DIR $ENV{MKL_ROOT_DIR}/mkl/lib/intel64)
set (MKL_LIB_DIR2 $ENV{MKL_ROOT_DIR}/compiler/lib/intel64)
set (MKL_BIN $ENV{MKL_ROOT_DIR}/redist/intel64/compiler)

set (MKL_LIBS "${MKL_LIB_DIR}/mkl_intel_ilp64.lib"
              "${MKL_LIB_DIR}/mkl_sequential.lib"
              "${MKL_LIB_DIR}/mkl_core.lib"
              "${MKL_LIB_DIR2}/libiomp5md.lib")

# Include sub-projects.
#add_subdirectory(tensorflow)
add_subdirectory(Utils)
add_subdirectory(Import)
add_subdirectory(Topology)
add_subdirectory(PolygonTriangularization)
add_subdirectory(MeshSegmentation)
add_subdirectory(splm)
add_subdirectory(Test)



