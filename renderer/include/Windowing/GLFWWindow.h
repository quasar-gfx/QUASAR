#ifndef GLFW_WINDOW_H
#define GLFW_WINDOW_H

#include <GLFW/glfw3.h>

#include <Windowing/Window.h>
#include <OpenGLAppConfig.h>

class GLFWWindow final : public Window {
public:
    GLFWwindow* window;

    explicit GLFWWindow(const Config &config);
    ~GLFWWindow() {
        glfwTerminate();
    }

    void getSize(unsigned int &width, unsigned int &height) override;

    bool resized() override;

    Mouse getMouseButtons() override;
    CursorPos getCursorPos() override;
    Keys getKeys() override;
    void setMouseCursor(bool enabled) override;
    void swapBuffers() override;

    double getTime() override;
    bool tick() override;

    void close() override;

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->frameResized = true;
    }

private:
    bool frameResized = true;
};

#endif // GLFW_WINDOW_H

