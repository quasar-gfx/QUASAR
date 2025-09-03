cmake_minimum_required(VERSION 3.22)
project(QUASAR)

add_subdirectory(simulator)
add_subdirectory(receiver)
add_subdirectory(streamer)

file(CREATE_LINK ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)
file(CREATE_LINK ${RENDERER_SHADERS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/shaders SYMBOLIC)
