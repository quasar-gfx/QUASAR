cmake_minimum_required(VERSION 3.22)
project(OpenGLRemoteRendering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

option(BUILD_APPS "Build apps" ON)

set(RENDERER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/renderer)
set(RENDERER_SHADERS_DIR ${RENDERER_DIR}/shaders)
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)

# include directories
include_directories(
    # renderer-related includes
    ${RENDERER_DIR}/include

    # for all external includes
    ${EXTERNAL_DIR}

    # for GLFW
    ${EXTERNAL_DIR}/glfw/include
)

# set required libs
set(RENDERER_LIBS )

if (UNIX)
    set(RENDERER_LIBS ${RENDERER_LIBS} ${CMAKE_DL_LIBS})
endif()

# add third party libraries
# glad
if (NOT ANDROID)
    add_library(glad ${EXTERNAL_DIR}/glad/glad.c)
    set(RENDERER_LIBS ${RENDERER_LIBS} glad)
endif()

# glfw
if (NOT ANDROID)
    set(GLFW_BUILD_EXAMPLES OFF)
    set(GLFW_BUILD_TESTS OFF)
    set(GLFW_BUILD_DOCS OFF)
    add_subdirectory(${EXTERNAL_DIR}/glfw)
    set(RENDERER_LIBS ${RENDERER_LIBS} glfw)
endif()

# imgui
if (NOT ANDROID)
    add_subdirectory(${EXTERNAL_DIR}/imgui/tnt)
    set(RENDERER_LIBS ${RENDERER_LIBS} imgui)
endif()

# spdlog
add_subdirectory(${EXTERNAL_DIR}/spdlog)
set(RENDERER_LIBS ${RENDERER_LIBS} spdlog)

# assimp
add_subdirectory(${EXTERNAL_DIR}/libassimp/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} assimp)

# stb
add_subdirectory(${EXTERNAL_DIR}/stb/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} stb)

# jsmn
add_subdirectory(${EXTERNAL_DIR}/jsmn/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} jsmn)

# lz4
add_subdirectory(${EXTERNAL_DIR}/lz4_stream/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} lz4_stream)

# zstd
set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_BUILD_SHARED OFF)
set(ZSTD_BUILD_TESTS OFF)
add_subdirectory(${EXTERNAL_DIR}/zstd/build/cmake)
set(RENDERER_LIBS ${RENDERER_LIBS} libzstd_static)

# add and link renderer library
add_subdirectory(${RENDERER_DIR})
target_link_libraries(renderer
    PUBLIC
    ${RENDERER_LIBS}
)

# add apps
if (BUILD_APPS)
    add_subdirectory(apps)

    # make symbolic link of assets to output/apps directory
    file(CREATE_LINK ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apps/assets SYMBOLIC)

    # make symbolic link of shaders to output/apps directory
    file(CREATE_LINK ${RENDERER_SHADERS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apps/shaders SYMBOLIC)
endif()
