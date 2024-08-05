cmake_minimum_required(VERSION 3.22)
project(Apps)

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

add_subdirectory(Common)

add_subdirectory(scene_viewer)
add_subdirectory(atw)
add_subdirectory(meshwarp)
add_subdirectory(quadwarp)
add_subdirectory(meshwarp2)
