cmake_minimum_required(VERSION 3.22)
set(TARGET quasar_common)
project(${TARGET})

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

set(LIBS )

set(LIBS ${LIBS} renderer)

# Threads
find_package(Threads REQUIRED)
set(LIBS ${LIBS} Threads::Threads)

# CUDA
if (NOT APPLE)
    find_package(CUDAToolkit REQUIRED)
    set(LIBS ${LIBS} ${CUDA_LIBRARIES})
    set(LIBS ${LIBS} CUDA::cudart)
    set(LIBS ${LIBS} CUDA::cuda_driver)
endif()

# ZLIB
find_package(ZLIB REQUIRED)
set(LIBS ${LIBS} ZLIB::ZLIB)

# FFmpeg
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

find_path(SWRESAMPLE_INCLUDE_DIR libswresample/swresample.h)
find_library(SWRESAMPLE_LIBRARY swresample)
set(LIBS ${LIBS} ${SWRESAMPLE_LIBRARY})

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
set(LIBS ${LIBS} ${GSTREAMER_LIBRARIES} ${GSTREAMER_APP_LIBRARIES})

file(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_library(${TARGET} STATIC ${PUBLIC_HDRS} ${SRCS})

target_compile_options(${TARGET} PRIVATE -Wno-cast-calling-convention -Wunused-variable)

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR} ${CMAKE_CURRENT_BINARY_DIR}/shaders
    ${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER_APP_INCLUDE_DIRS})

add_subdirectory(${SHADERS_DIR})

target_link_libraries(${TARGET} ${LIBS})

add_dependencies(${TARGET} shaders_common)
