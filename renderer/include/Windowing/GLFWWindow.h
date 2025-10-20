#ifndef GLFW_WINDOW_H
#define GLFW_WINDOW_H

#ifndef PLATFORM_ANDROID

#include <OpenGLAppConfig.h>
#include <Windowing/Window.h>

#include <GLFW/glfw3.h>

namespace quasar {

class GLFWWindow final : public Window {
public:
    GLFWwindow* window;

    GLFWWindow(const Config& config);
    ~GLFWWindow();

    glm::uvec2 getSize() override;

    bool resized() override;

    Mouse getMouseButtons() override;
    CursorPos getCursorPos() override;
    Keys getKeys() override;
    void setMouseCursor(bool enabled) override;
    ScrollOffset getScrollOffset() override;

    double getTime() override;
    bool tick() override;

    void swapBuffers() override;

    void close() override;

private:
    bool windowShouldClose = false;
    bool frameResized = true;

    ScrollOffset scrollOffset = { 0.0, 0.0 };

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto* me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->frameResized = true;
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        auto* me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->scrollOffset = {xoffset, yoffset};
    }
};

} // namespace quasar

#endif

#endif // GLFW_WINDOW_H
