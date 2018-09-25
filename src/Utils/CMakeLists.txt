
include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${PROJECT_SOURCE_DIR}/tensorflow/include
  C:/Users/marco/OneDrive/Documents/PROJECTS/ThirdParties/boost_1_66_0
)

file(GLOB HXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.hh")
file(GLOB CXX_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library (Utils STATIC ${CXX_FILES} ${HXX_FILES})


