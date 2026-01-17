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

# Add third party libraries
add_subdirectory(${EXTERNAL_DIR})

add_subdirectory(${RENDERER_DIR})
target_link_libraries(quasar_renderer PUBLIC quasar_external)

# Create target library
add_library(${TARGET} INTERFACE)
target_link_libraries(${TARGET} INTERFACE quasar_renderer)

# Add apps
if(QUASAR_BUILD_APPS)
    add_subdirectory(apps)

    # make symbolic link of assets to output/apps directory
    file(CREATE_LINK ${ASSETS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apps/assets SYMBOLIC)

    # make symbolic link of shaders to output/apps directory
    file(CREATE_LINK ${RENDERER_SHADERS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apps/shaders SYMBOLIC)
endif()
