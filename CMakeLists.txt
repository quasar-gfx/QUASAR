cmake_minimum_required(VERSION 3.19)
project(VideoStreaming)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(COMMON ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(LIBS )

find_package(OpenGL REQUIRED)
set(LIBS ${LIBS} ${OPENGL_LIBRARIES})

find_package(glfw3 REQUIRED)
set(LIBS ${LIBS} glfw)

add_library(GLAD "${EXTERNAL}/glad/glad.c")
set(LIBS ${LIBS} GLAD)

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)
set(LIBS ${LIBS} ${AVCODEC_LIBRARY})

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)
set(LIBS ${LIBS} ${AVFORMAT_LIBRARY})

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)
set(LIBS ${LIBS} ${AVUTIL_LIBRARY})

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)
set(LIBS ${LIBS} ${AVDEVICE_LIBRARY})

find_path(SWS_INCLUDE_DIR libswscale/swscale.h)
find_library(SWS_LIBRARY swscale)
set(LIBS ${LIBS} ${SWS_LIBRARY})

add_subdirectory(${EXTERNAL}/libassimp/tnt)
set(LIBS ${LIBS} assimp)

add_subdirectory(${EXTERNAL}/imgui/tnt)
set(LIBS ${LIBS} imgui)

file(GLOB COMMON_SRCS
    "${COMMON}/src/*.cpp"
    "${COMMON}/src/*.c"
)

include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR}
    ${COMMON}/include
    ${EXTERNAL}
)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/assets" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(streamer)
add_subdirectory(receiver)
