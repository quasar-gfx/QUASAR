cmake_minimum_required(VERSION 3.22)
project(ATW)

add_subdirectory(streamer)
add_subdirectory(receiver)
add_subdirectory(simulator)

file(CREATE_LINK ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)
file(CREATE_LINK ${RENDERER_SHADERS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/shaders SYMBOLIC)
