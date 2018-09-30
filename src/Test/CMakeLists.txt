# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#

set(name Test)

project(${name})

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hxx")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cxx")

# Add source to this project's executable.
add_executable(${name} ${CXX_FILES} ${HXX_FILES})
target_link_libraries(${name} 
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/lib/tensorflow.lib"
  Import Topology Geo PolygonTriangularization)
target_include_directories(${name} 
  PUBLIC ${CMAKE_SOURCE_DIR}/tensorflow/include
  PUBLIC ${CMAKE_SOURCE_DIR}/Catch2/single_include/Catch2)
  
target_compile_definitions(${name} PRIVATE NOMINMAX EIGEN_DEFAULT_DENSE_INDEX_TYPE=std::int64_t)

add_custom_command(
        TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll
                $<TARGET_FILE_DIR:${name}>/tensorflow.dll)

target_compile_options(${name}  PRIVATE /W4 /w44265 /WX- )

add_definitions(-DOUTDIR="${CMAKE_SOURCE_DIR}/../out")
add_definitions(-DINDIR="${CMAKE_SOURCE_DIR}/Test/Data")
add_definitions(-DPROTOBUF_USE_DLLS)

# TODO: Add tests and install targets if needed.
