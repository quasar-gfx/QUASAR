cmake_minimum_required(VERSION 3.22)
project(Apps)

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

set(EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(
    ${EXTERNAL}
)

add_subdirectory(Common)

add_subdirectory(scene_viewer)
add_subdirectory(atw)
add_subdirectory(meshwarp)
add_subdirectory(quadwarp)
add_subdirectory(meshwarp2)
