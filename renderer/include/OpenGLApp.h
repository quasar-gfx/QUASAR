#ifndef OPENGL_APP_H
#define OPENGL_APP_H

#include <glad/glad.h>

#include <functional>

#include <OpenGLRenderer.h>
#include <OpenGLAppConfig.h>
#include <Windowing/Window.h>

class OpenGLApp {
public:
    using RenderCallback = std::function<void(double now, double dt)>;
    using ResizeCallback = std::function<void(unsigned int width, unsigned int height)>;

    explicit OpenGLApp() = default;
    ~OpenGLApp() = default;

    Config config;

    OpenGLRenderer renderer;

    void init();
    void run();

    void onRender(RenderCallback callback) { renderCallback = callback; };
    void onResize(ResizeCallback callback) { resizeCallback = callback; };

    void resize(unsigned int width, unsigned int height);

private:
    RenderCallback renderCallback;
    ResizeCallback resizeCallback;
};

#endif // OPENGL_APP_H
