#ifndef OPENGL_APP_H
#define OPENGL_APP_H

#include <glad/glad.h>

#include <functional>

#include <OpenGLRenderer.h>
#include <OpenGLAppConfig.h>
#include <Window.h>

class OpenGLApp {
public:
    using GuiCallback = std::function<void(double now, double dt)>;
    using RenderCallback = std::function<void(double now, double dt)>;

    explicit OpenGLApp() = default;
    ~OpenGLApp() = default;

    Config config;

    OpenGLRenderer renderer;

    Window* window;

    void init(Window* window);
    void run();

    void gui(GuiCallback callback) { guiCallback = callback; }
    void onRender(RenderCallback callback) { renderCallback = callback; };

private:
    GuiCallback guiCallback;
    RenderCallback renderCallback;
};

#endif // OPENGL_APP_H
