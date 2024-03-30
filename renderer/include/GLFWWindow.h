#ifndef GLFW_WINDOW_H
#define GLFW_WINDOW_H

#include <iostream>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <Window.h>
#include <OpenGLAppConfig.h>

class GLFWWindow final : public Window {
public:
    GLFWWindow(const Config &config) {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, config.openglMajorVersion);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, config.openglMinorVersion);
        glfwWindowHint(GLFW_SAMPLES, config.numSamples);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

        window = glfwCreateWindow(config.width, config.height, config.title.c_str(), NULL, NULL);
        if (window == NULL) {
            throw std::runtime_error("Failed to create GLFW window");
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(config.enableVSync); // set vsync

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        ImGui::StyleColorsDark();

        // Setup ImGui OpenGL backend
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 410");

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return;
        }
    }

    ~GLFWWindow() {
        glfwTerminate();
    }

    void getSize(unsigned int* width, unsigned int* height) override {
        int frameBufferWidth, frameBufferHeight;
        glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
        while (frameBufferWidth == 0 || frameBufferHeight == 0) {
            glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
            glfwWaitEvents();
        }
        *width = frameBufferWidth;
        *height = frameBufferHeight;
    }

    bool resized() override {
        if (frameResized) {
            frameResized = false;
            return true;
        }
        return false;
    }

    Mouse getMouseButtons() override {
        Mouse mouse{
            .LEFT_PRESSED = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS),
            .MIDDLE_PRESSED = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS),
            .RIGHT_PRESSED = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        };
        return mouse;
    }

    CursorPos getCursorPos() override {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        CursorPos pos{
            .x = xpos,
            .y = ypos
        };
        return pos;
    }

    Keys getKeys() override {
        Keys keys {
            .W_PRESSED = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS),
            .A_PRESSED = (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS),
            .S_PRESSED = (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS),
            .D_PRESSED = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS),
            .ESC_PRESSED = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        };
        return keys;
    }

    void setMouseCursor(bool enabled) override {
        glfwSetInputMode(window, GLFW_CURSOR, enabled ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    }

    void swapBuffers() override {
        glfwSwapBuffers(window);
    }

    double getTime() override {
        return glfwGetTime();
    }

    bool tick() override {
        glfwPollEvents();
        return !glfwWindowShouldClose(window);
    }

    void close() override {
        glfwSetWindowShouldClose(window, true);
    }

    void guiNewFrame() override {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
    }

    void guiRender() override {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        auto me = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        me->frameResized = true;
    }

private:
    GLFWwindow* window;
    bool frameResized = false;
};

#endif // GLFW_WINDOW_H

