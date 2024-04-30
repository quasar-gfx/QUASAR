cmake_minimum_required(VERSION 3.19)
project(VideoStreaming)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

set(RENDERER ${CMAKE_CURRENT_SOURCE_DIR}/renderer)
set(SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# set required libs
set(RENDERER_LIBS )

set(RENDERER_LIBS ${RENDERER_LIBS} ${CMAKE_DL_LIBS})

find_package(OpenGL REQUIRED)
set(RENDERER_LIBS ${RENDERER_LIBS} ${OPENGL_LIBRARIES})

find_package(glfw3 REQUIRED)
set(RENDERER_LIBS ${RENDERER_LIBS} glfw)

add_library(glad "${EXTERNAL}/glad/glad.c")
set(RENDERER_LIBS ${RENDERER_LIBS} glad)

# add third party libraries
add_subdirectory(${EXTERNAL}/libassimp/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} assimp)

add_subdirectory(${EXTERNAL}/imgui/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} imgui)

add_subdirectory(${EXTERNAL}/stb/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} stb)

add_subdirectory(${EXTERNAL}/jsmn/tnt)
set(RENDERER_LIBS ${RENDERER_LIBS} jsmn)

add_subdirectory(${SHADERS})

# include directories
include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${RENDERER}/include
    ${EXTERNAL}
    ${CMAKE_BINARY_DIR}/shaders
)

# add and link renderer library
add_subdirectory(${RENDERER})
add_dependencies(renderer shaders)
target_link_libraries(renderer ${RENDERER_LIBS})

# add apps
add_subdirectory(apps)

# make symbolic link of assets to output/apps directory
file(CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_CURRENT_BINARY_DIR}/apps/assets
        SYMBOLIC
)

# make symbolic link of shaders to output/apps directory
file(CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        ${CMAKE_CURRENT_BINARY_DIR}/apps/shaders
        SYMBOLIC
)
