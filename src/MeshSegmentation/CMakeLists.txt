# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#

project(MeshSegmentation)

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hxx")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cxx")

# Add source to this project's executable.
add_executable(MeshSegmentation ${CXX_FILES} ${HXX_FILES})
target_link_libraries(MeshSegmentation 
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/lib/tensorflow.lib"
  Import Topology Geo PolygonTriangularization ${MKL_LIBS})
target_include_directories(MeshSegmentation PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/include")
  
add_custom_command(
        TARGET MeshSegmentation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll
                $<TARGET_FILE_DIR:MeshSegmentation>/tensorflow.dll)

add_definitions(-DOUTDIR="${CMAKE_SOURCE_DIR}/../out")
add_definitions(-DINDIR="${CMAKE_SOURCE_DIR}/Test/Data")
add_definitions(-DPROTOBUF_USE_DLLS)

# TODO: Add tests and install targets if needed.
