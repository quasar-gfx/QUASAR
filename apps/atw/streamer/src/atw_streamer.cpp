#include <iostream>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Utils/Utils.h>

#include <VideoStreamer.h>
#include <PoseReceiver.h>

int main(int argc, char** argv) {
    Config config{};
    config.title = "ATW Streamer";
    config.targetFramerate = 30;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> videoFormatIn(parser, "video-format", "Video format", {'g', "video-format"}, "mpegts");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "0.0.0.0:54321");
    args::ValueFlag<int> targetBitrateIn(parser, "targetBitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 50);
    args::ValueFlag<bool> vrModeIn(parser, "vr", "Enable VR mode", {'r', "vr"}, false);
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);
    config.showWindow = args::get(displayIn);

    std::string scenePath = args::get(scenePathIn);
    std::string videoURL = args::get(videoURLIn);
    std::string videoFormat = args::get(videoFormatIn);
    std::string poseURL = args::get(poseURLIn);

    unsigned int targetBitrate = args::get(targetBitrateIn);
    bool vrMode = args::get(vrModeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    Scene scene;
    std::unique_ptr<Camera> camera;
    SceneLoader loader;
    if (vrMode) {
        auto vrCamera = std::make_unique<VRCamera>(windowSize.x / 2, windowSize.y);
        loader.loadScene(scenePath, scene, vrCamera->left);
        vrCamera->right.setViewMatrix(vrCamera->left.getViewMatrix());
        vrCamera->right.setProjectionMatrix(vrCamera->left.getProjectionMatrix());
        camera = std::move(vrCamera);
    }
    else {
        auto perspectiveCamera = std::make_unique<PerspectiveCamera>(windowSize.x, windowSize.y);
        loader.loadScene(scenePath, scene, *perspectiveCamera);
        camera = std::move(perspectiveCamera);
    }

    glm::vec3 initialPosition = camera->getPosition();

    VideoStreamer videoStreamerRT = VideoStreamer({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL, config.targetFramerate, targetBitrate, videoFormat);

    PoseReceiver poseReceiver = PoseReceiver(camera.get(), poseURL);

    // shaders
    ToneMapShader toneMapShader;

    bool paused = false;
    RenderStats renderStats;
    pose_id_t currentFramePoseID;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";

        ImGui::NewFrame();

        unsigned int flags = 0;
        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &showFPS);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Frame Capture", 0, &showCaptureWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        if (showFPS) {
            ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
            flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
            ImGui::Begin("", 0, flags);
            ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
            ImGui::End();
        }

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %d", renderStats.drawCalls);

            ImGui::Separator();

            ImGui::Text("Video URL: %s (%s)", videoURL.c_str(), videoFormat.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamerRT.getFrameRate(), 1000.0f / videoStreamerRT.getFrameRate());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: %.3f ms", videoStreamerRT.stats.timeToCopyFrameMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: %.3f ms", videoStreamerRT.stats.timeToEncodeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: %.3f ms", videoStreamerRT.stats.timeToSendMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: %.3f Mbps", videoStreamerRT.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::Text("Remote Pose ID: %d", currentFramePoseID);

            ImGui::Separator();

            ImGui::Checkbox("Pause", &paused);

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                saveRenderTargetToFile(renderer, toneMapShader, fileName, saveAsHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        if (vrMode) {
            auto vrCamera = static_cast<VRCamera*>(camera.get());
            vrCamera->left.setAspect(windowSize.x / 2, windowSize.y);
            vrCamera->right.setAspect(windowSize.x / 2, windowSize.y);
            vrCamera->updateProjectionMatrix();
        }
        else {
            auto perspectiveCamera = static_cast<PerspectiveCamera*>(camera.get());
            perspectiveCamera->setAspect(windowSize.x, windowSize.y);
            perspectiveCamera->updateProjectionMatrix();
        }
    });

    std::vector<glm::vec3> pointLightPositions(4);
    pointLightPositions[0] = scene.pointLights[0]->position;
    pointLightPositions[1] = scene.pointLights[1]->position;
    pointLightPositions[2] = scene.pointLights[2]->position;
    pointLightPositions[3] = scene.pointLights[3]->position;

    app.onRender([&](double now, double dt) {
        // handle keyboard input
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (paused) {
            return;
        }

        // receive pose
        pose_id_t poseID = poseReceiver.receivePose();

        // animate lights
        // scene.pointLights[0]->setPosition(pointLightPositions[0] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        // scene.pointLights[1]->setPosition(pointLightPositions[1] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        // scene.pointLights[2]->setPosition(pointLightPositions[2] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        // scene.pointLights[3]->setPosition(pointLightPositions[3] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));

        // offset camera
        if (camera->isVR()) {
            auto* vrCamera = static_cast<VRCamera*>(camera.get());
            vrCamera->left.setPosition(vrCamera->left.getPosition() + initialPosition);
            vrCamera->right.setPosition(vrCamera->right.getPosition() + initialPosition);
            vrCamera->left.updateViewMatrix();
            vrCamera->right.updateViewMatrix();
        }
        else {
            auto* perspectiveCamera = static_cast<PerspectiveCamera*>(camera.get());
            perspectiveCamera->setPosition(perspectiveCamera->getPosition() + initialPosition);
            perspectiveCamera->updateViewMatrix();
        }

        renderer.drawObjects(scene, *camera);

        // restore camera position
        if (camera->isVR()) {
            auto* vrCamera = static_cast<VRCamera*>(camera.get());
            vrCamera->left.setPosition(vrCamera->left.getPosition() - initialPosition);
            vrCamera->right.setPosition(vrCamera->right.getPosition() - initialPosition);
            vrCamera->left.updateViewMatrix();
            vrCamera->right.updateViewMatrix();
        }
        else {
            auto* perspectiveCamera = static_cast<PerspectiveCamera*>(camera.get());
            perspectiveCamera->setPosition(perspectiveCamera->getPosition() - initialPosition);
            perspectiveCamera->updateViewMatrix();
        }

        // copy rendered result to video render target
        renderer.drawToRenderTarget(toneMapShader, videoStreamerRT);
        if (config.showWindow) {
            renderer.drawToScreen(toneMapShader);
        }

        // send video frame
        if (poseID != -1) {
            currentFramePoseID = poseID;
            videoStreamerRT.sendFrame(poseID);
        }
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
