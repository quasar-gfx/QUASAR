#ifndef PLATFORM_H
#define PLATFORM_H

#if defined(__ANDROID__)
    #define PLATFORM_ANDROID
    #define GL_ES
    #include <GLES3/gl32.h>
#elif defined(__APPLE__)
    #define PLATFORM_APPLE
    #define GL_CORE
    #include <glad/glad.h>
#elif defined(_WIN32) || defined(_WIN64)
    #define PLATFORM_WINDOWS
    #define GL_CORE
    #include <glad/glad.h>
#elif defined(__linux__)
    #define PLATFORM_LINUX
    #define GL_CORE
    #include <glad/glad.h>
#endif

// type aliases for convenience
#ifndef uint
    typedef unsigned int uint;
#endif
#ifndef uchar
    typedef unsigned char uchar;
#endif
#ifndef ushort
    typedef unsigned short ushort;
#endif
#ifndef ulong
    typedef unsigned long ulong;
#endif
#ifndef int8
    typedef signed char int8;
#endif
#ifndef uint8
    typedef unsigned char uint8;
#endif
#ifndef int16
    typedef short int16;
#endif
#ifndef uint16
    typedef unsigned short uint16;
#endif
#ifndef int32
    typedef int int32;
#endif
#ifndef uint32
    typedef unsigned int uint32;
#endif
#ifndef int64
    typedef long long int64;
#endif
#ifndef uint64
    typedef unsigned long long uint64;
#endif

#endif // PLATFORM_H
