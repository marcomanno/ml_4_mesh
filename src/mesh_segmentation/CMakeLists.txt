# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#

project(mesh_segmentation)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hxx")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cxx")

# Add source to this project's executable.
add_executable(mesh_segmentation ${CXX_FILES} ${HXX_FILES})
target_link_libraries(mesh_segmentation 
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/lib/tensorflow.lib")
target_include_directories(mesh_segmentation PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/include")
  
target_compile_definitions(mesh_segmentation PRIVATE NOMINMAX EIGEN_DEFAULT_DENSE_INDEX_TYPE=std::int64_t)

add_custom_command(
        TARGET mesh_segmentation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll
                ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.dll)

# TODO: Add tests and install targets if needed.
