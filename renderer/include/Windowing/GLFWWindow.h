#ifndef GLFW_WINDOW_H
#define GLFW_WINDOW_H

#ifndef __ANDROID__

#include <Windowing/Window.h>
#include <OpenGLAppConfig.h>

#include <GLFW/glfw3.h>

class GLFWWindow final : public Window {
public:
    GLFWwindow* window;

    GLFWWindow(const Config &config);
    ~GLFWWindow() {
        glfwTerminate();
    }

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

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto* me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->frameResized = true;
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        auto* me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->scrollOffset = {xoffset, yoffset};
    }

private:
    ScrollOffset scrollOffset = {0.0, 0.0};

    bool frameResized = true;
};

#endif

#endif // GLFW_WINDOW_H
