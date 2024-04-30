#ifndef OPENGL_APP_CONFIG_H
#define OPENGL_APP_CONFIG_H

#include <string>

struct Config {
    bool enableVSync = true;
    bool showWindow = true;
    bool sRGB = true;
    bool backfaceCulling = false;
    unsigned char openglMajorVersion = 4;
    unsigned char openglMinorVersion = 1;
    unsigned char numSamples = 4;
    unsigned int width = 800;
    unsigned int height = 600;
    std::string title = "OpenGL App";
};

#endif // OPENGL_APP_CONFIG_H
