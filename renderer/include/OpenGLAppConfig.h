#ifndef OPENGL_APP_CONFIG_H
#define OPENGL_APP_CONFIG_H

#include <string>
#include <memory>

#include <GraphicsPipeline.h>
#include <Windowing/Window.h>
#include <GUI/GUIManager.h>

namespace quasar {

struct Config {
    bool enableVSync = true;
    bool showWindow = true;
    unsigned char openglMajorVersion = 4;
#ifndef __APPLE__
    unsigned char openglMinorVersion = 6;
#else
    unsigned char openglMinorVersion = 1;
#endif
    GraphicsPipeline pipeline;
    unsigned int width = 800;
    unsigned int height = 600;
    unsigned int targetFramerate = 60;
    std::string title = "OpenGL App";
    std::shared_ptr<Window> window = nullptr;
    std::shared_ptr<GUIManager> guiManager = nullptr;
};

} // namespace quasar

#endif // OPENGL_APP_CONFIG_H
