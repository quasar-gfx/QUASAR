#include <iostream>
#include <thread>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Primitives/Entity.h>
#include <OpenGLApp.h>

OpenGLApp::OpenGLApp(const Config &config)
        : window(config.window)
        , guiManager(config.guiManager)
        , targetFramerate(config.enableVSync ? config.targetFramerate : 0) {
    // check version
    if (config.openglMajorVersion < 3 || (config.openglMajorVersion == 3 && config.openglMinorVersion < 3)) {
        throw std::runtime_error("OpenGL version must be 3.3 or higher");
    }
#ifdef __APPLE__
    if (config.openglMajorVersion == 4 && config.openglMinorVersion > 1) {
        throw std::runtime_error("OpenGL version cannot be higher than 4.1 on MacOS");
    }
#endif
}

void OpenGLApp::run() {
    const double targetDeltaTime = targetFramerate != 0 ? 1.0 / targetFramerate : 0.0;

    double prevTime = window->getTime();
    while (window->tick()) {
        double currTime = window->getTime();
        double deltaTime = currTime - prevTime;

        if (deltaTime < targetDeltaTime) {
            // sleep for the remaining time to meet target framerate
            std::this_thread::sleep_for(std::chrono::duration<double>(targetDeltaTime - deltaTime));
            currTime = window->getTime();
            deltaTime = currTime - prevTime;
        }

        if (window->resized()) {
            glm::uvec2 windowSize = window->getSize();

            std::cout << "Resized to " << windowSize.x << "x" << windowSize.y << std::endl;

            if (resizeCallback) {
                resizeCallback(windowSize.x, windowSize.y);
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
