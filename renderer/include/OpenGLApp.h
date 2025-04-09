#ifndef OPENGL_APP_H
#define OPENGL_APP_H

#include <functional>

#include <OpenGLAppConfig.h>
#include <Windowing/Window.h>

namespace quasar {

class OpenGLApp {
public:
    using RenderCallback = std::function<void(double now, double dt)>;
    using ResizeCallback = std::function<void(unsigned int width, unsigned int height)>;

    OpenGLApp(const Config &config);
    ~OpenGLApp() = default;

    void onRender(RenderCallback callback) { renderCallback = callback; };
    void onResize(ResizeCallback callback) { resizeCallback = callback; };

    virtual void run();

private:
    double targetFramerate;

    RenderCallback renderCallback;
    ResizeCallback resizeCallback;

    std::shared_ptr<Window> window;
    std::shared_ptr<GUIManager> guiManager;
};

} // namespace quasar

#endif // OPENGL_APP_H
