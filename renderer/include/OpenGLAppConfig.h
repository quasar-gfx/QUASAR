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
    bool sortTransparent = true;
    unsigned char openglMajorVersion = 4;
#ifndef PLATFORM_APPLE
    unsigned char openglMinorVersion = 6;
#else
    unsigned char openglMinorVersion = 1;
#endif
    GraphicsPipeline pipeline;
    uint width = 800;
    uint height = 600;
    uint targetFramerate = 60;
    std::string title = "OpenGL App";
    std::shared_ptr<Window> window = nullptr;
    std::shared_ptr<GUIManager> guiManager = nullptr;
};

} // namespace quasar

#endif // OPENGL_APP_CONFIG_H
