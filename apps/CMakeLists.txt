cmake_minimum_required(VERSION 3.19)
project(Apps)

include_directories(
    common
)

add_subdirectory(test)
add_subdirectory(atw)
add_subdirectory(meshwarp)
add_subdirectory(quadwarp)
add_subdirectory(meshwarp2)
