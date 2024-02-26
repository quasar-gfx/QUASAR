#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <OpenGLApp.h>

int OpenGLApp::init() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(config.width, config.height, config.title.c_str(), NULL, NULL);
    if (window == NULL) {
        throw std::runtime_error("Failed to create GLFW window");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (config.enableVSync) {
        glfwSwapInterval(1); // Enable vsync
    }

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseMoveCallbackWrapper);
    glfwSetScrollCallback(window, mouseScrollCallbackWrapper);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    // Setup ImGui OpenGL backend
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // set default camera parameters
    int width, height;
    getWindowSize(&width, &height);
    camera.setProjectionMatrix(glm::radians(60.0f), (float)width / (float)height, 0.1f, 100.0f);
    camera.position = glm::vec3(0.0f, 1.6f, 2.0f);

    return 0;
}

void OpenGLApp::cleanup() {
    glfwTerminate();
}

void OpenGLApp::run() {
    float currTime;
    float prevTime = static_cast<float>(glfwGetTime());
    while (!glfwWindowShouldClose(window)) {
        currTime = static_cast<float>(glfwGetTime());
        float deltaTime = currTime - prevTime;

        glfwPollEvents();

        if (frameResized) {
            int width, height;
            getWindowSize(&width, &height);
            std::cout << "Resized to " << width << "x" << height << std::endl;
            camera.aspect = (float)width / (float)height;
            glViewport(0, 0, width, height);
            frameResized = false;
        }

        camera.updateViewMatrix();
        camera.updateProjectionMatrix();

        if (animCallback) {
            animCallback(currTime, deltaTime);
        }

        if (guiCallback) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            guiCallback(currTime, deltaTime);
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(window);

        prevTime = currTime;
    }
}
