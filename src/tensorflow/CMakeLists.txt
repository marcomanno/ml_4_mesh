# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
project (TensorFlow)

add_library(TensorFlow INTERFACE)

target_link_libraries(TensorFlow "${CMAKE_CURRENT_SOURCE_DIR}/lib/tensorflow_static.lib")

set_target_properties(TensorFlow PROPERTIES
    LINK_INTERFACE_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/lib/tensorflow_static.lib")

set_target_properties(TensorFlow PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
