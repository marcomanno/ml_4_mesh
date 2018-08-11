
project(Import)

include_directories (
  ..
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# collect all header and source files
file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hh")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(PolygonTriangularization STATIC ${CXX_FILES} ${HXX_FILES})


