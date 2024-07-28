#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Primatives/Entity.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>

OpenGLApp::OpenGLApp(const Config &config) : window(config.window), guiManager(config.guiManager) {
    // enable face culling
    if (config.backfaceCulling) {
        glEnable(GL_CULL_FACE);
    }

    // enable srgb framebuffers
    if (config.sRGB) {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }

    renderer = std::make_unique<OpenGLRenderer>(config.width, config.height);
}

void OpenGLApp::run() {
    double prevTime = window->getTime();
    while (window->tick()) {
        double currTime = window->getTime();
        double deltaTime = currTime - prevTime;

        if (window->resized()) {
            unsigned int width, height;
            window->getSize(width, height);

            std::cout << "Resized to " << width << "x" << height << std::endl;
            renderer->resize(width, height);

            if (resizeCallback) {
                resizeCallback(width, height);
            }
        }

        if (renderCallback) {
            renderCallback(currTime, deltaTime);
        }

        if (guiManager) {
            guiManager->draw(currTime, deltaTime);
        }

        window->swapBuffers();

        prevTime = currTime;
    }
}

void OpenGLApp::resize(unsigned int width, unsigned int height) {
    renderer->resize(width, height);
}
