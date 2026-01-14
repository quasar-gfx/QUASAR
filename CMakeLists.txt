cmake_minimum_required(VERSION 3.22)
set(TARGET quasar)
project(${TARGET})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QUASAR_BUILD_APPS "Build apps" ON)

set(RENDERER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/renderer)
set(RENDERER_SHADERS_DIR ${RENDERER_DIR}/shaders)
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)

# Set required libs
set(RENDERER_LIBS )

if(UNIX)
    set(RENDERER_LIBS ${RENDERER_LIBS} ${CMAKE_DL_LIBS})
endif()

# Add third party libraries
# glad
if(NOT ANDROID)
    add_library(glad ${EXTERNAL_DIR}/glad/glad.c)
    target_include_directories(glad PUBLIC ${EXTERNAL_DIR})
    set(RENDERER_LIBS ${RENDERER_LIBS} glad)
endif()

# glfw
if(NOT ANDROID)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${EXTERNAL_DIR}/glfw)
    set(RENDERER_LIBS ${RENDERER_LIBS} glfw)
endif()

# imgui
if(NOT ANDROID)
    add_subdirectory(${EXTERNAL_DIR}/imgui/tnt)
    # use bundled glfw
    target_link_libraries(imgui PUBLIC glfw)
    target_include_directories(imgui PUBLIC ${EXTERNAL_DIR}/glfw/include)
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

# lz4
add_subdirectory(${EXTERNAL_DIR}/lz4/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} lz4)

# zstd
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/zstd/build/cmake)
set(RENDERER_LIBS ${RENDERER_LIBS} libzstd_static)

# Add and link renderer library
add_subdirectory(${RENDERER_DIR})
target_include_directories(renderer PUBLIC ${EXTERNAL_DIR})
target_link_libraries(renderer PUBLIC ${RENDERER_LIBS})

# Create target library
add_library(${TARGET} INTERFACE)
target_include_directories(${TARGET} INTERFACE ${EXTERNAL_DIR})
target_link_libraries(${TARGET} INTERFACE renderer)

# Add apps
if(QUASAR_BUILD_APPS)
    add_subdirectory(apps)

    # make symbolic link of assets to output/apps directory
    file(CREATE_LINK ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apps/assets SYMBOLIC)

    # make symbolic link of shaders to output/apps directory
    file(CREATE_LINK ${RENDERER_SHADERS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apps/shaders SYMBOLIC)
endif()
