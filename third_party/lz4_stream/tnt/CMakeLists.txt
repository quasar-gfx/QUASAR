cmake_minimum_required(VERSION 3.19)
project(lz4_stream)

set(OUR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(TARGET      lz4_stream)
set(LZ4_LIB_DIR ${OUR_DIR}/lz4/lib)

set(LZ4_SRCS
    ${LZ4_LIB_DIR}/lz4.c
    ${LZ4_LIB_DIR}/lz4frame.c
    ${LZ4_LIB_DIR}/lz4hc.c
    ${LZ4_LIB_DIR}/xxhash.c
)

add_library(lz4 STATIC ${LZ4_SRCS})

target_include_directories(lz4 PUBLIC ${LZ4_LIB_DIR})

if(ANDROID)
    target_compile_options(lz4 PRIVATE -fPIC -DANDROID)
endif()

add_library(${TARGET} INTERFACE)

target_include_directories(${TARGET}
    INTERFACE
    ${OUR_DIR}
    ${LZ4_LIB_DIR}
)

target_link_libraries(${TARGET} INTERFACE lz4)

set_target_properties(${TARGET} PROPERTIES FOLDER ThirdParty)
