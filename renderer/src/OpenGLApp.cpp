#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <Entity.h>
#include <OpenGLApp.h>

unsigned int Entity::nextID = 0;
unsigned int Node::nextID = 0;

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
    glfwSwapInterval(config.enableVSync); // set vsync

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

    // // set default camera parameters
    // int width, height;
    // getWindowSize(&width, &height);
    // camera.setProjectionMatrix(glm::radians(60.0f), (float)width / (float)height, 0.1f, 100.0f);
    // camera.position = glm::vec3(0.0f, 1.6f, 2.0f);

    return 0;
}

void OpenGLApp::cleanup() {
    glfwTerminate();
}

void OpenGLApp::draw(Shader &shader, Scene* scene, Camera* camera) {
    glm::mat4 model;

    shader.bind();

    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());

    for (auto child : scene->children) {
        drawNode(shader, child, glm::mat4(1.0f));
    }

    shader.unbind();
}

void OpenGLApp::drawNode(Shader &shader, Node* node, glm::mat4 parentTransform) {
    glm::mat4 model = parentTransform * node->getTransformParentFromLocal();

    if (node->entity != nullptr) {
        shader.setMat4("model", model);
        node->entity->draw(shader);
    }

    for (auto& child : node->children) {
        drawNode(shader, child, model);
    }
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
            if (resizeCallback) {
                resizeCallback(width, height);
            }
            glViewport(0, 0, width, height);
            frameResized = false;
        }

        if (renderCallback) {
            renderCallback(currTime, deltaTime);
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
