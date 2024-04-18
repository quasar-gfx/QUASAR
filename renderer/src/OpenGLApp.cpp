#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Primatives/Entity.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>

void OpenGLApp::init(Window* window) {
    this->window = window;
    renderer.init(config.width, config.height);
}

void OpenGLApp::run() {
    double prevTime = window->getTime();
    while (window->tick()) {
        double currTime = window->getTime();
        double deltaTime = currTime - prevTime;

        if (window->resized()) {
            unsigned int width, height;
            window->getSize(&width, &height);
            std::cout << "Resized to " << width << "x" << height << std::endl;
            glViewport(0, 0, width, height);
        }

        if (renderCallback) {
            renderCallback(currTime, deltaTime);
        }

        if (guiCallback) {
            window->guiNewFrame();
            guiCallback(currTime, deltaTime);
            window->guiRender();
        }

        window->swapBuffers();

        prevTime = currTime;
    }
}
