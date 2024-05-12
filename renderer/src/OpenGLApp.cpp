#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Primatives/Entity.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>

void OpenGLApp::init() {
    // enable face culling
    if (config.backfaceCulling) {
        glEnable(GL_CULL_FACE);
    }

    // enable srgb framebuffers
    if (config.sRGB) {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }

    renderer.init(config.width, config.height);
}

void OpenGLApp::run() {
    double prevTime = config.window->getTime();
    while (config.window->tick()) {
        double currTime = config.window->getTime();
        double deltaTime = currTime - prevTime;

        if (config.window->resized()) {
            unsigned int width, height;
            config.window->getSize(&width, &height);

            std::cout << "Resized to " << width << "x" << height << std::endl;
            renderer.resize(width, height);

            if (resizeCallback) {
                resizeCallback(width, height);
            }
        }

        if (renderCallback) {
            renderCallback(currTime, deltaTime);
        }

        if (config.guiManager) {
            config.guiManager->draw(currTime, deltaTime);
        }

        config.window->swapBuffers();

        prevTime = currTime;
    }
}

void OpenGLApp::resize(unsigned int width, unsigned int height) {
    renderer.resize(width, height);
}
