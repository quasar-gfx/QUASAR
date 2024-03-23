#ifndef OPENGL_APP_H
#define OPENGL_APP_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <functional>

#include <OpenGLRenderer.h>
#include <OpenGLAppConfig.h>

class OpenGLApp {
public:
    using GuiCallback = std::function<void(double now, double dt)>;
    using ResizeCallback = std::function<void(unsigned int width, unsigned int height)>;
    using MouseMoveCallback = std::function<void(double xpos, double ypos)>;
    using MouseScrollCallback = std::function<void(double xoffset, double yoffset)>;
    using RenderCallback = std::function<void(double now, double dt)>;

    OpenGLApp() = default;
    ~OpenGLApp() = default;

    Config config;

    OpenGLRenderer renderer;

    GLFWwindow* window;
    bool frameResized = false;

    int init();
    void cleanup();
    void run();

    void gui(GuiCallback callback) { guiCallback = callback; }
    void onResize(ResizeCallback callback) { resizeCallback = callback; }
    void onMouseMove(MouseMoveCallback callback) { mouseMoveCallback = callback; }
    void onMouseScroll(MouseScrollCallback callback) { scrollCallback = callback; }
    void onRender(RenderCallback callback) { renderCallback = callback; };

    void getWindowSize(unsigned int *resWidth, unsigned int *resHeight) const {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        *resWidth = width;
        *resHeight = height;
    }

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<OpenGLApp*>(glfwGetWindowUserPointer(window));
        app->frameResized = true;
    }

private:
    GuiCallback guiCallback;
    ResizeCallback resizeCallback;
    MouseMoveCallback mouseMoveCallback;
    MouseScrollCallback scrollCallback;
    RenderCallback renderCallback;

    static void mouseMoveCallbackWrapper(GLFWwindow* window, double xpos, double ypos) {
        auto app = reinterpret_cast<OpenGLApp*>(glfwGetWindowUserPointer(window));
        if (app->mouseMoveCallback) {
            app->mouseMoveCallback(xpos, ypos);
        }
    }

    static void mouseScrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset) {
        auto app = reinterpret_cast<OpenGLApp*>(glfwGetWindowUserPointer(window));
        if (app->scrollCallback) {
            app->scrollCallback(xoffset, yoffset);
        }
    }
};

#endif // OPENGL_APP_H
