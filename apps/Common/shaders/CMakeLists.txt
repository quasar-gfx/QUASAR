cmake_minimum_required(VERSION 3.22)
project(ShadersCommon)

set(TARGET shaders_common)

set(EMBEDFILE_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/embedfile.py)

file(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.geom"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.comp"
)

set(SHADERS_HEADER "${CMAKE_CURRENT_BINARY_DIR}/shaders_common.h")

# delete old header file
add_custom_command(
    OUTPUT ${SHADERS_HEADER}
    COMMAND ${CMAKE_COMMAND} -E remove ${SHADERS_HEADER}
    DEPENDS ${SRCS})

foreach(GLSL_SRC ${SRCS})
    get_filename_component(FILE_NAME ${GLSL_SRC} NAME_WE)
    get_filename_component(FILE_EXT ${GLSL_SRC} LAST_EXT)
    string(SUBSTRING ${FILE_EXT} 1 -1 FILE_EXT)
    set(FILE_NAME "SHADER_COMMON_${FILE_NAME}_${FILE_EXT}")
    string(TOUPPER ${FILE_NAME} FILE_NAME_UPPER)

    set(TEMP_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.h")

    add_custom_command(
        OUTPUT ${SHADERS_HEADER}
        COMMAND ${EMBEDFILE_SCRIPT} ${FILE_NAME_UPPER} ${GLSL_SRC} ${SHADERS_HEADER}
        DEPENDS ${GLSL_SRC}
        APPEND)

    list(APPEND TEMP_HEADERS ${TEMP_HEADER})
endforeach(GLSL_SRC)

add_custom_target(${TARGET} DEPENDS ${SHADERS_HEADER})
