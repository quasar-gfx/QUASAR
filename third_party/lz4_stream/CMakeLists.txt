cmake_minimum_required(VERSION 3.19)
project(lz4_stream)

set(OUR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(TARGET lz4_stream)

set(LZ4_SRCS
    ${OUR_DIR}/lz4/lib/lz4.c
    ${OUR_DIR}/lz4/lib/lz4frame.c
    ${OUR_DIR}/lz4/lib/lz4hc.c
    ${OUR_DIR}/lz4/lib/xxhash.c
)

add_library(lz4 STATIC ${LZ4_SRCS})

target_include_directories(lz4 PUBLIC ${OUR_DIR}/lz4/lib)

if(ANDROID)
    target_compile_options(lz4 PRIVATE -fPIC -DANDROID)
endif()

add_library(${TARGET} INTERFACE)

target_include_directories(${TARGET}
    INTERFACE
    ${OUR_DIR}
    ${OUR_DIR}/lz4/lib
)

target_link_libraries(${TARGET} INTERFACE lz4)

set_target_properties(${TARGET} PROPERTIES FOLDER ThirdParty)
