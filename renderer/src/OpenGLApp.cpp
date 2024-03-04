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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
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

    // enable backface culling
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);

    // enable seamless cube map sampling for lower mip levels in the pre-filter map
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // enable msaa
    glEnable(GL_MULTISAMPLE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    return 0;
}

void OpenGLApp::cleanup() {
    glfwTerminate();
}

void OpenGLApp::drawSkyBox(Shader &shader, Scene* scene, Camera* camera) {
    shader.bind();
    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(shader);
    }

    if (scene->skyBox != nullptr) {
        scene->skyBox->draw(shader, camera);
    }

    shader.unbind();
}

void OpenGLApp::draw(Shader &shader, Scene* scene, Camera* camera) {
    shader.bind();

    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());
    shader.setVec3("viewPos", camera->position);

    if (scene->skyBox != nullptr) {
        glActiveTexture(GL_TEXTURE0 + 4);
        glBindTexture(GL_TEXTURE_CUBE_MAP, scene->skyBox->ID);
        shader.setInt("skybox", 4);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(shader);
    }

    if (scene->directionalLight != nullptr) {
        scene->directionalLight->draw(shader);
    }

    for (int i = 0; i < scene->pointLights.size(); i++) {
        scene->pointLights[i]->draw(shader, i);
    }

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
