#ifndef OPENGL_APP_H
#define OPENGL_APP_H

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <functional>

#include "Camera.h"
#include "OpenGLAppConfig.h"

class OpenGLApp {
public:
    using MouseMoveCallback = std::function<void(double xpos, double ypos)>;
    using MouseScrollCallback = std::function<void(double xoffset, double yoffset)>;
    using AnimCallback = std::function<void(double now, double dt)>;

    OpenGLApp() = default;
    ~OpenGLApp() = default;

    Config config;

    GLFWwindow* window;

    Camera camera;

    int init();
    void cleanup();
    void animate(AnimCallback callback) { animCallback = callback; };
    void run();

    void mouseMove(MouseMoveCallback callback) { mouseMoveCallback = callback; }
    void mouseScroll(MouseScrollCallback callback) { scrollCallback = callback; }

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<OpenGLApp*>(glfwGetWindowUserPointer(window));
        app->camera.aspect = (float)width / (float)height;
        glViewport(0, 0, width, height);
    }

private:
    MouseMoveCallback mouseMoveCallback;
    MouseScrollCallback scrollCallback;
    AnimCallback animCallback;

    static void mouseMoveCallbackWarpper(GLFWwindow* window, double xpos, double ypos) {
        auto app = reinterpret_cast<OpenGLApp*>(glfwGetWindowUserPointer(window));
        if (app->mouseMoveCallback) {
            app->mouseMoveCallback(xpos, ypos);
        }
    }

    static void mouseScrollCallbackWarpper(GLFWwindow* window, double xoffset, double yoffset) {
        auto app = reinterpret_cast<OpenGLApp*>(glfwGetWindowUserPointer(window));
        if (app->scrollCallback) {
            app->scrollCallback(xoffset, yoffset);
        }
    }
};

#endif // OPENGL_APP_H
