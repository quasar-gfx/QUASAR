cmake_minimum_required(VERSION 3.22)
project(OpenGLRenderer)

set(TARGET renderer)

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRCS
    "${RENDERER}/src/*.cpp"
    "${RENDERER}/src/*.c"
)

add_library(${TARGET} STATIC ${PUBLIC_HDRS} ${SRCS})

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR})
