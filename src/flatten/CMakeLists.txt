set(project_name flatten)
project(${project_name})

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  $ENV{MKL_ROOT_DIR}/mkl/include
  ${CMAKE_SOURCE_DIR}/tensorflow/include
  ${CMAKE_SOURCE_DIR}/eigen-git-mirror
  ${CMAKE_SOURCE_DIR}/spectra/include
)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hxx")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cxx")

add_library(${project_name} STATIC ${CXX_FILES} ${HXX_FILES})

target_link_libraries(${project_name} 
    PRIVATE Topology 
    PRIVATE Geo 
    PRIVATE splm_lib 
    PRIVATE ${MKL_LIBS})
