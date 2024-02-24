#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "OpenGLApp.h"

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
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseMoveCallbackWarpper);
    glfwSetScrollCallback(window, mouseScrollCallbackWarpper);

    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // set default camera parameters
    camera.setProjectionMatrix(glm::radians(60.0f), (float)config.width / (float)config.height, 0.1f, 100.0f);
    camera.position = glm::vec3(0.0f, 0.0f, 2.0f);

    return 0;
}

void OpenGLApp::cleanup() {
    glfwTerminate();
}

void OpenGLApp::run() {
    float currTime;
    float prevTime = static_cast<float>(glfwGetTime());
    while (!glfwWindowShouldClose(window)) {
        camera.updateViewMatrix();
        camera.updateProjectionMatrix();

        currTime = static_cast<float>(glfwGetTime());
        if (animCallback) {
            animCallback(currTime, currTime - prevTime);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();

        prevTime = currTime;
    }
}
