#ifndef GLFW_WINDOW_H
#define GLFW_WINDOW_H

#include <iostream>

#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <Window.h>
#include <OpenGLAppConfig.h>

class GLFWWindow final : public Window {
public:
    explicit GLFWWindow(const Config &config);
    ~GLFWWindow() {
        glfwTerminate();
    }

    void getSize(unsigned int* width, unsigned int* height) override;

    bool resized() override;

    Mouse getMouseButtons() override;
    CursorPos getCursorPos() override;
    Keys getKeys() override;
    void setMouseCursor(bool enabled) override;
    void swapBuffers() override;

    double getTime() override;
    bool tick() override;

    void guiNewFrame() override;
    void guiRender() override;

    void close() override;

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->frameResized = true;
    }

private:
    GLFWwindow* window;
    bool frameResized = false;
};

#endif // GLFW_WINDOW_H

