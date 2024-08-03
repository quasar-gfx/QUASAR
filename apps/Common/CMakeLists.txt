cmake_minimum_required(VERSION 3.22)
project(Common)

set(TARGET common)

set(PUBLIC_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBS )

set(LIBS ${LIBS} renderer)

find_package(Threads REQUIRED)
set(LIBS ${LIBS} Threads::Threads)

if (NOT APPLE)
    find_package(CUDAToolkit REQUIRED)
    set(LIBS ${LIBS} ${CUDA_LIBRARIES})
    set(LIBS ${LIBS} CUDA::cudart)
    set(LIBS ${LIBS} CUDA::cuda_driver)
endif()

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

file(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_library(${TARGET} STATIC ${PUBLIC_HDRS} ${SRCS})

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR})

target_link_libraries(${TARGET} ${LIBS})
