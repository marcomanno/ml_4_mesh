# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project (TensorFlowExperiments)

if(MSVC)
add_definitions(-D_USE_MATH_DEFINES)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest /W0")
add_definitions(-DMKL_ILP64)
add_definitions(-DMKL_DIRECT_CALL)
endif(MSVC)

set(EIGEN_INCLUDE_DIR "C:/Users/USER/source/repos/3p/Eigen")
SET(BUILD_TESTING OFF CACHE BOOL "Use some expat option")
SET(BUILD_BENCHMARKS OFF CACHE BOOL "Use some expat option 2")
set(MINIGLOG on)

# Include sub-projects.
#add_subdirectory(tensorflow)
add_subdirectory(sparselm)
add_subdirectory(Utils)
add_subdirectory(Geo)
add_subdirectory(Import)
add_subdirectory(Topology)
add_subdirectory(PolygonTriangularization)
add_subdirectory(MeshSegmentation)
add_subdirectory(Test)



