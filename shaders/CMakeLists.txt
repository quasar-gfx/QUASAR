cmake_minimum_required(VERSION 3.19)
project(shaders)

add_executable(embedfile ${CMAKE_SOURCE_DIR}/cmake/embedfile.c)

file(GLOB_RECURSE SRCS
    "*.frag"
    "*.vert"
    "*.geom"
    "*.comp"
)

set(SHADER_HEADER "${CMAKE_BINARY_DIR}/shaders/shaders.h")

# delete old header file
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E remove ${SHADER_HEADER}
    DEPENDS ${SRCS}
)

foreach(GLSL_SRC ${SRCS})
    get_filename_component(FILE_NAME ${GLSL_SRC} NAME_WE)
    get_filename_component(FILE_EXT ${GLSL_SRC} LAST_EXT)
    string(SUBSTRING ${FILE_EXT} 1 -1 FILE_EXT)
    set(FILE_NAME "SHADER_${FILE_NAME}_${FILE_EXT}")
    string(TOUPPER ${FILE_NAME} FILE_NAME_UPPER)

    set(TEMP_HEADER "${CMAKE_BINARY_DIR}/shaders/${FILE_NAME}.h")

    add_custom_command(
            OUTPUT ${SHADER_HEADER}
            COMMAND embedfile ${FILE_NAME_UPPER} ${GLSL_SRC} ${SHADER_HEADER}
            DEPENDS ${GLSL_SRC}
            APPEND)

    list(APPEND TEMP_HEADERS ${TEMP_HEADER})
endforeach(GLSL_SRC)

add_custom_target(shaders DEPENDS ${SHADER_HEADER})
