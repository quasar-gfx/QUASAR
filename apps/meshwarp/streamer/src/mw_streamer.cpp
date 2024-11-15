#include <iostream>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Utils/Utils.h>

#include <Shaders/ComputeShader.h>
#include <VideoStreamer.h>
#include <DepthStreamer.h>
#include <BC4DepthStreamer.h>
#include <PoseReceiver.h>

enum class PauseState {
    PLAY,
    PAUSE_COLOR,
    PAUSE_DEPTH,
    PAUSE_BOTH
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> videoFormatIn(parser, "video-format", "Video format", {'g', "video-format"}, "mpegts");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "0.0.0.0:54321");
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
    args::ValueFlag<int> targetBitrateIn(parser, "targetBitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 50);
    args::ValueFlag<int> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
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
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    unsigned int targetBitrate = args::get(targetBitrateIn);
    int depthFactor = args::get(depthFactorIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    SceneLoader loader;
    loader.loadScene(scenePath, scene, camera);

    // set fov
    camera.setFovyDegrees(args::get(fovIn));

    glm::vec3 initialPosition = camera.getPosition();

    VideoStreamer videoStreamerColorRT = VideoStreamer({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL, targetBitrate, videoFormat);

    BC4DepthStreamer BC4videoStreamerDepthRT = BC4DepthStreamer({
        .width = windowSize.x / depthFactor,
        .height = windowSize.y / depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    }, depthURL);

    PoseReceiver poseReceiver = PoseReceiver(&camera, poseURL);

    // shaders
    ToneMapShader toneMapShader;

    Shader depthShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYDEPTH_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYDEPTH_FRAG_len
    });

    PauseState pauseState = PauseState::PLAY;
    RenderStats renderStats;
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

            ImGui::Separator();

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
            ImGui::Text("Depth URL: %s", depthURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f fps), BC4 D (%.1f fps)", videoStreamerColorRT.getFrameRate(), BC4videoStreamerDepthRT.getFrameRate());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: RGB (%.3f ms), BC4 D (%.3f ms)", videoStreamerColorRT.stats.timeToCopyFrameMs, BC4videoStreamerDepthRT.stats.timeToCopyFrameMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: RGB (%.3f ms), BC4 D (%.3f ms)", videoStreamerColorRT.stats.timeToSendMs, BC4videoStreamerDepthRT.stats.timeToSendMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: RGB (%.3f Mbps), BC4 D (%.3f Mbps)", videoStreamerColorRT.stats.bitrateMbps, BC4videoStreamerDepthRT.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::RadioButton("Play All", (int*)&pauseState, 0);
            ImGui::RadioButton("Pause Color", (int*)&pauseState, 1);
            ImGui::RadioButton("Pause Depth", (int*)&pauseState, 2);
            ImGui::RadioButton("Pause Both", (int*)&pauseState, 3);

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
                saveRenderTargetToFile(renderer, toneMapShader, fileName, windowSize, saveAsHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.resize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    std::vector<glm::vec3> pointLightPositions(4);
    pointLightPositions[0] = scene.pointLights[0]->position;
    pointLightPositions[1] = scene.pointLights[1]->position;
    pointLightPositions[2] = scene.pointLights[2]->position;
    pointLightPositions[3] = scene.pointLights[3]->position;

    pose_id_t poseID = 0;
    app.onRender([&](double now, double dt) {
        if (pauseState == PauseState::PAUSE_BOTH) {
            return;
        }

        // receive pose
        poseID = poseReceiver.receivePose(false);

        // animate lights
        // scene.pointLights[0]->setPosition(pointLightPositions[0] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        // scene.pointLights[1]->setPosition(pointLightPositions[1] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        // scene.pointLights[2]->setPosition(pointLightPositions[2] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        // scene.pointLights[3]->setPosition(pointLightPositions[3] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));

        // offset camera
        camera.setPosition(camera.getPosition() + initialPosition);
        camera.updateViewMatrix();

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // restore camera position
        camera.setPosition(camera.getPosition() - initialPosition);
        camera.updateViewMatrix();

        renderer.drawToRenderTarget(toneMapShader, videoStreamerColorRT);
        depthShader.bind();
        depthShader.setFloat("near", camera.getNear());
        depthShader.setFloat("far", camera.getFar());
        renderer.drawToRenderTarget(depthShader, BC4videoStreamerDepthRT);

        // render to screen
        if (config.showWindow) {
            renderer.drawToScreen(toneMapShader);
        }

        // Send compressed depth frame
        if (poseID != -1) {
            if (pauseState != PauseState::PAUSE_COLOR) videoStreamerColorRT.sendFrame(poseID);
            if (pauseState != PauseState::PAUSE_DEPTH) BC4videoStreamerDepthRT.sendFrame(poseID);
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
