#ifndef PLATFORM_H
#define PLATFORM_H

#if defined(__ANDROID__)
#include <GLES3/gl31.h>
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__) || defined(_WIN32) || defined(_WIN64)
#include <glad/glad.h>
#endif

#endif // PLATFORM_H
