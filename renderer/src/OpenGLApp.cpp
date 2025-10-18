#include <spdlog/spdlog.h>

#include <Primitives/Entity.h>
#include <OpenGLApp.h>

using namespace quasar;

OpenGLApp::OpenGLApp(const Config& config)
    : window(config.window)
    , guiManager(config.guiManager)
    , targetFramerate(config.enableVSync ? config.targetFramerate : 0)
{
    spdlog::set_pattern("[%H:%M:%S] [%^%L%$] %v");

    // Check opengl version
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
            // Sleep for the remaining time to meet target framerate
            std::this_thread::sleep_for(std::chrono::duration<double>(targetDeltaTime - deltaTime));
            currTime = window->getTime();
            deltaTime = currTime - prevTime;
        }

        if (window->resized()) {
            glm::uvec2 windowSize = window->getSize();
            if (resizeCallback) {
                resizeCallback(windowSize.x, windowSize.y);
            }

            spdlog::info("Window resized to {}x{}", windowSize.x, windowSize.y);
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
