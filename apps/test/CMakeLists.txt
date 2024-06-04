cmake_minimum_required(VERSION 3.19)
project(TestApp)

set(TARGET test)

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBS )

set(LIBS ${LIBS} common)

file(GLOB SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/shaders" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(${TARGET} ${SRCS})

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR})

target_link_libraries(${TARGET} PRIVATE ${LIBS})
