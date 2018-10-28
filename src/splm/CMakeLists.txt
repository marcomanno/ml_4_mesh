set(project_name splm_lib)

project(${project_name})

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hxx")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cxx")

add_library(${project_name} STATIC ${CXX_FILES} ${HXX_FILES})

target_include_directories(${project_name}
  PRIVATE $ENV{MKL_ROOT_DIR}/mkl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/tensorflow/include)

