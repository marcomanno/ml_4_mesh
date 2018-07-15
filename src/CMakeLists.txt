# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project (TensorFlowExperiments)

# Include sub-projects.
#add_subdirectory(tensorflow)
add_subdirectory(mesh_segmentation)
