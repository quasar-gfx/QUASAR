#ifndef OPENGL_APP_H
#define OPENGL_APP_H

#include <functional>

#include <OpenGLRenderer.h>
#include <OpenGLAppConfig.h>
#include <Windowing/Window.h>

class OpenGLApp {
public:
    using RenderCallback = std::function<void(double now, double dt)>;
    using ResizeCallback = std::function<void(unsigned int width, unsigned int height)>;

    explicit OpenGLApp(const Config &config);
    ~OpenGLApp() = default;

    std::unique_ptr<OpenGLRenderer> renderer;

    void run();

    void onRender(RenderCallback callback) { renderCallback = callback; };
    void onResize(ResizeCallback callback) { resizeCallback = callback; };

    void resize(unsigned int width, unsigned int height);

private:
    RenderCallback renderCallback;
    ResizeCallback resizeCallback;

    std::shared_ptr<Window> window;
    std::shared_ptr<GUIManager> guiManager;
};

#endif // OPENGL_APP_H
