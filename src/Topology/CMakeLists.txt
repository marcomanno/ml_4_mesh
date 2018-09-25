project(Topology)

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${PROJECT_SOURCE_DIR}/tensorflow/include
)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hh")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library (Topology STATIC ${CXX_FILES} ${HXX_FILES})


