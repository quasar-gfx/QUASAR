cmake_minimum_required(VERSION 3.19)
project(TestApp)

set(TARGET test)

file(GLOB SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/shaders" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(${TARGET} ${RENDERER_SRCS} ${SRCS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(${TARGET} PRIVATE ${LIBS})
