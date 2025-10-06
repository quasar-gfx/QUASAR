#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/DeferredRenderer.h>
#include <PostProcessing/Tonemapper.h>

#include <Streamers/MeshWarpStreamer.h>
#include <Receivers/PoseReceiver.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Streamer";
    config.targetFramerate = 30;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<int> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
    args::ValueFlag<int> targetBitrateIn(parser, "target-bitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 12);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "URL to send video", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "URL to send depth", {'e', "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "URL to send camera pose", {'p', "pose-url"}, "0.0.0.0:54321");
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

    if (verbose) spdlog::set_level(spdlog::level::debug);

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);
    config.showWindow = args::get(displayIn);

    Path sceneFile = args::get(sceneFileIn);
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    uint targetBitrate = args::get(targetBitrateIn);
    int depthFactor = args::get(depthFactorIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    DeferredRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    float remoteFOV = args::get(fovIn);
    camera.setFovyDegrees(remoteFOV);

    glm::vec3 initialPosition = camera.getPosition();

    MeshWarpStreamer meshWarpStreamer(
        renderer, scene, camera,
        videoURL, depthURL,
        depthFactor,
        config.targetFramerate, targetBitrate);
    PoseReceiver poseReceiver(&camera, poseURL);

    Tonemapper tonemapper;

    bool sendFrame = true;

    const double serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = 5; // default to 30 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    RenderStats renderStats;
    pose_id_t prevPoseID;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;

        ImGui::NewFrame();

        ImGuiWindowFlags flags = 0;
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

            glm::vec3 position = camera.getPosition();
            glm::vec3 rotation = camera.getRotationEuler();
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Camera Position", (float*)&position);
            ImGui::DragFloat3("Camera Rotation", (float*)&rotation);
            ImGui::EndDisabled();

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Depth URL: %s", depthURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::Text("Client Pose ID: %d", prevPoseID);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f fps), BC4 D (%.1f fps)",
                                                    meshWarpStreamer.getVideoFrameRate(),
                                                    meshWarpStreamer.getDepthFrameRate());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: RGB (%.3f ms), BC4 D (%.3f ms)",
                                                    meshWarpStreamer.videoStreamerRT.stats.transferTimeMs,
                                                    meshWarpStreamer.depthStreamerRT.stats.transferTimeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: RGB (%.3f ms), BC4 D (%.3f ms)",
                                                    meshWarpStreamer.videoStreamerRT.stats.encodeTimeMs,
                                                    meshWarpStreamer.depthStreamerRT.stats.compressTimeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: RGB (%.3f ms), BC4 D (%.3f ms)",
                                                    meshWarpStreamer.videoStreamerRT.stats.sendTimeMs,
                                                    meshWarpStreamer.depthStreamerRT.stats.sendTimeMs);

            ImGui::Separator();

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            if (ImGui::Button("Send Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                sendFrame = true;
            }

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });

    double totalDT = 0.0;
    double lastRenderTime = -INFINITY;
    app.onRender([&](double now, double dt) {
        // Handle keyboard input
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }
        totalDT += dt;

        if (rerenderIntervalMs > 0.0 && (now - lastRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendFrame = true;
        }
        if (sendFrame) {
            // Update all animations
            scene.updateAnimations(totalDT);
            totalDT = 0.0;
            lastRenderTime = now;

            // Receive pose
            pose_id_t poseID = poseReceiver.receivePose();
            if (poseID != -1 && poseID != prevPoseID) {
                // Offset camera
                camera.setPosition(camera.getPosition() + initialPosition);
                camera.updateViewMatrix();

                renderStats = meshWarpStreamer.generateFrame();
                meshWarpStreamer.sendFrame(poseID);

                // Restore camera position
                camera.setPosition(camera.getPosition() - initialPosition);
                camera.updateViewMatrix();

                // Send video and depth frames
                prevPoseID = poseID;
            }

            sendFrame = false;
        }

        // Render to screen
        if (config.showWindow) {
            tonemapper.drawToScreen(renderer);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
