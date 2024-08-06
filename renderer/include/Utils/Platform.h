#ifndef PLATFORM_H
#define PLATFORM_H

#if defined(__ANDROID__)
#define GL_ES
#include <GLES3/gl32.h>
#elif defined(__linux__) || defined(_WIN32) || defined(_WIN64)
#define GL_CORE
#include <glad/glad.h>
#elif defined(__APPLE__)
#define GL_CORE
#include <glad/glad.h>
#endif

#endif // PLATFORM_H
