cmake_minimum_required(VERSION 3.22)
set(TARGET quasar_common)
project(${TARGET})

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

set(LIBS )

set(LIBS ${LIBS} quasar_renderer)

# Threads
find_package(Threads REQUIRED)
set(LIBS ${LIBS} Threads::Threads)

# CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA compiler is available, but CUDAToolkit not found. Disabling CUDA support.")
        set(QUASAR_HAS_CUDA 0)
    else()
        message(STATUS "CUDAToolkit found. Enabling CUDA support.")
        set(LIBS ${LIBS} CUDA::cudart CUDA::cuda_driver)
        set(QUASAR_HAS_CUDA 1)
    endif()
else()
    message(STATUS "CUDA not found. Disabling CUDA support.")
    set(QUASAR_HAS_CUDA 0)
endif()

# ZLIB
find_package(ZLIB REQUIRED)
set(LIBS ${LIBS} ZLIB::ZLIB)

find_package(PkgConfig REQUIRED)

# FFmpeg
# pkg-config provides FFmpeg includes and transitive dependencies
pkg_check_modules(FFMPEG IMPORTED_TARGET
    libavcodec libavformat libavutil libavdevice libswscale libswresample)
if(FFMPEG_FOUND)
    set(LIBS ${LIBS} PkgConfig::FFMPEG)
else()
    # Require FFmpeg components when pkg-config metadata is unavailable
    find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h REQUIRED)
    find_library(AVCODEC_LIBRARY avcodec REQUIRED)
    find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h REQUIRED)
    find_library(AVFORMAT_LIBRARY avformat REQUIRED)
    find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h REQUIRED)
    find_library(AVUTIL_LIBRARY avutil REQUIRED)
    find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h REQUIRED)
    find_library(AVDEVICE_LIBRARY avdevice REQUIRED)
    find_path(SWS_INCLUDE_DIR libswscale/swscale.h REQUIRED)
    find_library(SWS_LIBRARY swscale REQUIRED)
    find_path(SWRESAMPLE_INCLUDE_DIR libswresample/swresample.h REQUIRED)
    find_library(SWRESAMPLE_LIBRARY swresample REQUIRED)
    set(LIBS ${LIBS}
        ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY}
        ${AVDEVICE_LIBRARY} ${SWS_LIBRARY} ${SWRESAMPLE_LIBRARY}
    )
endif()

# GStreamer
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
set(LIBS ${LIBS} PkgConfig::GSTREAMER PkgConfig::GSTREAMER_APP)
# The above code may add duplicate libraries, so remove them
list(REMOVE_DUPLICATES LIBS)

file(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_library(${TARGET} STATIC ${PUBLIC_HDRS} ${SRCS})

target_compile_options(${TARGET} PRIVATE -Wno-cast-calling-convention -Wunused-variable)
if(QUASAR_HAS_CUDA)
    target_compile_options(${TARGET} PUBLIC -DQUASAR_HAS_CUDA)
endif()

target_include_directories(${TARGET} PUBLIC
    ${PUBLIC_HDR_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/shaders
)
if(NOT FFMPEG_FOUND)
    target_include_directories(${TARGET} PUBLIC
        ${AVCODEC_INCLUDE_DIR}
        ${AVFORMAT_INCLUDE_DIR}
        ${AVUTIL_INCLUDE_DIR}
        ${AVDEVICE_INCLUDE_DIR}
        ${SWS_INCLUDE_DIR}
        ${SWRESAMPLE_INCLUDE_DIR}
    )
endif()

add_subdirectory(${SHADERS_DIR})

target_link_libraries(${TARGET} ${LIBS})

add_dependencies(${TARGET} shaders_common)
