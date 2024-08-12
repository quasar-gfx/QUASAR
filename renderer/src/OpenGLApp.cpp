#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Primatives/Entity.h>
#include <OpenGLApp.h>

OpenGLApp::OpenGLApp(const Config &config) : window(config.window), guiManager(config.guiManager) {
    // check version
    if (config.openglMajorVersion < 3 || (config.openglMajorVersion == 3 && config.openglMinorVersion < 3)) {
        throw std::runtime_error("OpenGL version must be 3.3 or higher");
    }
#ifdef __APPLE__
    if (config.openglMajorVersion == 4 && config.openglMinorVersion > 1) {
        throw std::runtime_error("OpenGL version cannot be higher than 4.1 on MacOS");
    }
#endif

    renderer = std::make_unique<DepthPeelingRenderer>(config.width, config.height);
    renderer->setGraphicsPipeline(config.pipeline);
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
