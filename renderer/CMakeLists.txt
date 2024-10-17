cmake_minimum_required(VERSION 3.22)
project(OpenGLRenderer)

set(TARGET renderer)

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

file(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_library(${TARGET} STATIC ${PUBLIC_HDRS} ${SRCS})

add_subdirectory(${SHADERS_DIR})

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR} ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_dependencies(${TARGET} shaders_builtin)
