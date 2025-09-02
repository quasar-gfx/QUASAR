cmake_minimum_required(VERSION 3.22)
project(QuadStreamReceiver)

set(TARGET qs_receiver)

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBS )

set(LIBS ${LIBS} quasar_common)

file(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders SYMBOLIC)

add_executable(${TARGET} ${SRCS})

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR})

target_link_libraries(${TARGET} PRIVATE ${LIBS})

target_compile_options(${TARGET} PRIVATE -Wno-cast-calling-convention -Wunused-variable)
