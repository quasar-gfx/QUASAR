cmake_minimum_required(VERSION 3.22)
set(TARGET quasar_external)
project(${TARGET})

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBS )

if(UNIX)
    set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
endif()

# glad
if(NOT ANDROID)
    add_library(glad ${EXTERNAL_DIR}/glad/glad.c)
    target_include_directories(glad PUBLIC ${EXTERNAL_DIR})
    set(LIBS ${LIBS} glad)
endif()

# glfw
if(NOT ANDROID)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${EXTERNAL_DIR}/glfw)
    set(LIBS ${LIBS} glfw)
endif()

# imgui
if(NOT ANDROID)
    add_subdirectory(${EXTERNAL_DIR}/imgui/tnt)
    if(UNIX)
        target_compile_options(imgui PRIVATE -Wno-deprecated-enum-enum-conversion)
    endif()
    # use bundled glfw
    target_link_libraries(imgui PUBLIC glfw)
    target_include_directories(imgui PUBLIC ${EXTERNAL_DIR}/glfw/include)
    set(LIBS ${LIBS} imgui)
endif()

# spdlog
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "")
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "")
set(SPDLOG_INSTALL OFF CACHE BOOL "")
add_subdirectory(${EXTERNAL_DIR}/spdlog)
set(LIBS ${LIBS} spdlog)

# assimp
set(ARGS_BUILD_EXAMPLE OFF CACHE BOOL "")
set(ARGS_BUILD_UNITTESTS OFF CACHE BOOL "")
add_subdirectory(${EXTERNAL_DIR}/libassimp/tnt)
set(LIBS ${LIBS} assimp)

# stb
add_subdirectory(${EXTERNAL_DIR}/stb/tnt)
set(LIBS ${LIBS} stb)

# Imath
set(IMATH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IMATH_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/Imath)
if(ANDROID)
    target_compile_options(Imath PRIVATE -Wnopointer-sign)
endif()
set(LIBS ${LIBS} Imath::Imath)

# openexr
set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/openexr)
if(UNIX)
    target_compile_options(OpenEXR PRIVATE -Wno-deprecated-declarations)
endif()
if(ANDROID)
    target_compile_options(OpenEXRCore PRIVATE -Wno-pointer-sign -Wno-format)
endif()
set(LIBS ${LIBS} OpenEXR::OpenEXR)

# lz4
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/lz4/build/cmake)
set(LIBS ${LIBS} lz4_static)

# zstd
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/zstd/build/cmake)
set(LIBS ${LIBS} libzstd_static)

# Create target library
add_library(${TARGET} INTERFACE)
target_include_directories(${TARGET} INTERFACE ${EXTERNAL_DIR})
target_link_libraries(${TARGET} INTERFACE ${LIBS})
