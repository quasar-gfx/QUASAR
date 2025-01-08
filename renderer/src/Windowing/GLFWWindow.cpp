#include <iostream>

#include <spdlog/spdlog.h>

#include <Windowing/GLFWWindow.h>

GLFWWindow::GLFWWindow(const Config &config) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, config.openglMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, config.openglMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, config.pipeline.multiSampleState.numSamples);
    glfwWindowHint(GLFW_VISIBLE, config.showWindow);

    window = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);
    if (window == nullptr) {
        throw std::runtime_error("Failed to create GLFW window");
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(config.enableVSync); // set vsync

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetScrollCallback(window, scrollCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        spdlog::error("Failed to initialize GLAD");
        return;
    }
}

glm::uvec2 GLFWWindow::getSize() {
    int frameBufferWidth, frameBufferHeight;
    glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
    while (frameBufferWidth == 0 || frameBufferHeight == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
    }

    return glm::vec2(frameBufferWidth, frameBufferHeight);
}

bool GLFWWindow::resized() {
    if (frameResized) {
        frameResized = false;
        return true;
    }
    return false;
}

Mouse GLFWWindow::getMouseButtons() {
    Mouse mouse{
        .LEFT_PRESSED = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS),
        .MIDDLE_PRESSED = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS),
        .RIGHT_PRESSED = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    };
    return mouse;
}

CursorPos GLFWWindow::getCursorPos() {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    CursorPos pos{
        .x = xpos,
        .y = ypos
    };
    return pos;
}

Keys GLFWWindow::getKeys() {
    Keys keys {
        .W_PRESSED = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS),
        .A_PRESSED = (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS),
        .S_PRESSED = (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS),
        .D_PRESSED = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS),
        .Q_PRESSED = (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS),
        .E_PRESSED = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS),
        .ESC_PRESSED = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    };
    return keys;
}

ScrollOffset GLFWWindow::getScrollOffset() {
    auto res = scrollOffset;
    scrollOffset = {0.0, 0.0};
    return res;
}

void GLFWWindow::setMouseCursor(bool enabled) {
    glfwSetInputMode(window, GLFW_CURSOR, enabled ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

void GLFWWindow::swapBuffers() {
    glfwSwapBuffers(window);
}

double GLFWWindow::getTime() {
    return glfwGetTime();
}

bool GLFWWindow::tick() {
    glfwPollEvents();
    return !glfwWindowShouldClose(window);
}

void GLFWWindow::close() {
    glfwSetWindowShouldClose(window, true);
}
