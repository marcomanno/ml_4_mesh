# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#

set(project_name Test)

project(${project_name})

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hxx")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cxx")

# Add source to this project's executable.
add_executable(${project_name} ${CXX_FILES} ${HXX_FILES})
target_link_libraries(${project_name} 
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/lib/tensorflow.lib"
  Import Topology Geo PolygonTriangularization)
target_include_directories(${project_name}
  PUBLIC ${CMAKE_SOURCE_DIR}/tensorflow/include
  PUBLIC ${CMAKE_SOURCE_DIR}/CppNumericalSolvers/include
  PUBLIC ${CMAKE_SOURCE_DIR}/Catch2/single_include/Catch2)
  
add_custom_command(
        TARGET ${project_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll
                $<TARGET_FILE_DIR:${project_name}>/tensorflow.dll)

target_compile_options(${project_name}  PRIVATE /W3 /WX- /DMKL_ILP64 /Oi)

target_link_libraries(${project_name}
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/lib/tensorflow.lib"
  Import Topology Geo PolygonTriangularization splm_lib ${MKL_LIBS})


add_definitions(-DOUTDIR="${CMAKE_SOURCE_DIR}/../out")
add_definitions(-DINDIR="${CMAKE_SOURCE_DIR}/Test/Data")

# TODO: Add tests and install targets if needed.
add_custom_command(TARGET ${project_name}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                           ${MKL_BIN}/libiomp5md.dll  ${CMAKE_CURRENT_BINARY_DIR}/libiomp5md.dll
                   VERBATIM)
