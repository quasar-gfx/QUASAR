#ifndef OPENGL_APP_CONFIG_H
#define OPENGL_APP_CONFIG_H

#include <string>
#include <memory>

#include <Windowing/Window.h>
#include <GUI/GUIManager.h>

struct Config {
    bool enableVSync = true;
    bool showWindow = true;
    bool sRGB = true;
    bool backfaceCulling = false;
    unsigned char openglMajorVersion = 4;
#ifndef __APPLE__
    unsigned char openglMinorVersion = 6;
#else
    unsigned char openglMinorVersion = 1;
#endif
    unsigned char numSamples = 4;
    unsigned int width = 800;
    unsigned int height = 600;
    std::string title = "OpenGL App";
    std::shared_ptr<Window> window = nullptr;
    std::shared_ptr<GUIManager> guiManager = nullptr;
};

#endif // OPENGL_APP_CONFIG_H
