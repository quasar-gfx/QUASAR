#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Framebuffer.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <VideoTexture.h>
#include <PoseStreamer.h>

int main(int argc, char** argv) {
    Config config{};
    config.title = "ATW Receiver";
    config.sRGB = false;

    std::string videoURL = "127.0.0.1:1234";
    std::string poseURL = "127.0.0.1:4321";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            videoURL = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            poseURL = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
    }

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

    VideoTexture videoTexture({
        .width = config.width,
        .height = config.height,
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    });
    videoTexture.initVideo(videoURL);
    PoseStreamer poseStreamer(&camera, poseURL);

    std::cout << "Video URL: " << videoURL << std::endl;
    std::cout << "Pose URL: " << poseURL << std::endl;

    bool atwEnabled = false;
    double elapedTime = 0.0f;
    guiManager->onRender([&](double now, double dt) {
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        int flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("", 0, flags);
        ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);
        glm::vec2 guiSize = winSize * glm::vec2(0.4f, 0.3f);
        ImGui::SetNextWindowSize(ImVec2(guiSize.x, guiSize.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 60), ImGuiCond_FirstUseEver);
        flags = 0;
        ImGui::Begin(config.title.c_str(), 0, flags);
        ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

        ImGui::Separator();

        ImGui::Text("Video URL: %s", videoURL.c_str());
        ImGui::Text("Pose URL: %s", poseURL.c_str());

        ImGui::Separator();

        ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoTexture.getFrameRate(), 1000.0f / videoTexture.getFrameRate());
        ImGui::TextColored(ImVec4(1,0.5,0,1), "E2E Latency: %.3f ms", elapedTime * 1000.0f);

        ImGui::Separator();

        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms", videoTexture.stats.timeToReceiveFrame);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decode frame: %.3f ms", videoTexture.stats.timeToDecode);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to resize frame: %.3f ms", videoTexture.stats.timeToResize);

        ImGui::Separator();

        ImGui::Checkbox("ATW Enabled", &atwEnabled);

        ImGui::End();
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    // shaders
    Shader screenShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "shaders/displayVideo.frag"
    });

    camera.position = glm::vec3(0.0f, 1.6f, 0.0f);

    Pose currentFramePose;
    app.onRender([&](double now, double dt) {
        // handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;
            if (!prevMouseLeftPressed && mouseButtons.LEFT_PRESSED) {
                dragging = true;
                prevMouseLeftPressed = true;

                auto cursorPos = window->getCursorPos();
                lastX = static_cast<float>(cursorPos.x);
                lastY = static_cast<float>(cursorPos.y);
            }
            if (prevMouseLeftPressed && !mouseButtons.LEFT_PRESSED) {
                dragging = false;
                prevMouseLeftPressed = false;
            }
            if (dragging) {
                auto cursorPos = window->getCursorPos();
                float xpos = static_cast<float>(cursorPos.x);
                float ypos = static_cast<float>(cursorPos.y);

                float xoffset = xpos - lastX;
                float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

                lastX = xpos;
                lastY = ypos;

                camera.processMouseMovement(xoffset, yoffset, true);
            }

            // handle keyboard input
            auto keys = window->getKeys();
            camera.processKeyboard(keys, dt);
            if (keys.ESC_PRESSED) {
                window->close();
            }
        }

        // send pose to streamer
        poseStreamer.sendPose(now);

        {
            screenShader.bind();

            screenShader.setBool("atwEnabled", atwEnabled);

            glm::mat4 proj = camera.getProjectionMatrix();
            glm::mat4 view = camera.getViewMatrix();
            screenShader.setMat4("projection", proj);
            screenShader.setMat4("view", view);
            screenShader.setInt("videoTexture", 5);
            videoTexture.bind(5);
            // render video frame
            pose_id_t poseId = videoTexture.draw();

            if (poseId != -1 && poseStreamer.getPose(poseId, &currentFramePose, now, &elapedTime)) {
                screenShader.setMat4("remoteProjection", currentFramePose.proj);
                screenShader.setMat4("remoteView", currentFramePose.view);
            }
        }

        // render to screen
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    videoTexture.cleanup();

    return 0;
}
