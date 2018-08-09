
project(Import)

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hh")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library (Import STATIC ${CXX_FILES} ${HXX_FILES})


