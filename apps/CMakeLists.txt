cmake_minimum_required(VERSION 3.19)
project(Apps)

include_directories(
    common
)

add_subdirectory(test)
add_subdirectory(streamer)
add_subdirectory(receiver)
add_subdirectory(meshing)
add_subdirectory(meshingviz)
add_subdirectory(quads)
add_subdirectory(quadsviz)
