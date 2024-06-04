cmake_minimum_required(VERSION 3.19)
project(MeshWarp2)

add_subdirectory(streamer)
add_subdirectory(receiver)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/../../assets ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/../../shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders SYMBOLIC)
