#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>
#include <PostProcessing/ToneMapper.h>

#include <Streamers/QuadsStreamer.h>
#include <Receivers/PoseReceiver.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Streamer";
    config.targetFramerate = 30;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 60.0f);
    args::ValueFlag<int> targetBitrateIn(parser, "targetBitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 15);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "URL to send video", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> proxiesURLIn(parser, "proxies", "URL to send quad proxy metadata", {'e', "proxies-url"}, "127.0.0.1:65432");
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

    // Parse remote size
    std::string rsizeStr = args::get(resIn);
    pos = rsizeStr.find('x');
    glm::uvec2 remoteWindowSize = glm::uvec2(std::stoi(rsizeStr.substr(0, pos)), std::stoi(rsizeStr.substr(pos + 1)));

    config.enableVSync = !args::get(novsync);
    config.showWindow = args::get(displayIn);

    Path sceneFile = args::get(sceneFileIn);
    std::string videoURL = args::get(videoURLIn);
    std::string proxiesURL = args::get(proxiesURLIn);
    std::string poseURL = args::get(poseURLIn);

    uint targetBitrate = args::get(targetBitrateIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DeferredRenderer remoteRenderer(config);

    // "Remote" scene
    Scene scene;
    PerspectiveCamera camera(remoteRenderer.width, remoteRenderer.height);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    float remoteFOV = args::get(remoteFOVIn);
    camera.setFovyDegrees(remoteFOV);

    glm::vec3 initialPosition = camera.getPosition();

    QuadSet quadSet(remoteWindowSize);
    QuadsStreamer quadwarp(
        quadSet,
        remoteRenderer, scene, camera,
        videoURL, proxiesURL,
        targetBitrate);

    // "Local" scene for visualization
    Scene localScene;
    quadwarp.addMeshesToScene(localScene);

    PoseReceiver poseReceiver(&camera, poseURL);

    // Post processing
    ToneMapper toneMapper;

    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;

    bool sendReferenceFrame = true;
    bool sendResidualFrame = false;
    int refFrameInterval = 2;

    const int serverFPSValues[] = {0, 1, 2, 3, 4, 5};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "2 FPS", "3 FPS", "4 FPS", "5 FPS"};
    int serverFPSIndex = 1; // default to 1 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showFramePreviewWindow = false;

        static bool showSkyBox = true;

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
            ImGui::MenuItem("Frame Previews", 0, &showFramePreviewWindow);
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

            uint totalTriangles = quadwarp.getNumTriangles();
            if (totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %d", totalTriangles);
            else if (totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %d", totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %d", totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %d", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               quadwarp.stats.totalSizes.numQuads,
                               quadwarp.stats.totalSizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quadwarp.stats.totalSizes.numDepthOffsets,
                               quadwarp.stats.totalSizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            glm::vec3 position = camera.getPosition();
            glm::vec3 rotation = camera.getRotationEuler();
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Camera Position", (float*)&position);
            ImGui::DragFloat3("Camera Rotation", (float*)&rotation);
            ImGui::EndDisabled();

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Proxies URL: %s", proxiesURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Background Settings")) {
                if (ImGui::Checkbox("Show Sky Box", &showSkyBox)) {
                    localScene.envCubeMap = showSkyBox ? scene.envCubeMap : nullptr;
                }

                if (ImGui::Button("Change Background Color", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    ImGui::OpenPopup("Background Color Popup");
                }
                if (ImGui::BeginPopup("Background Color Popup")) {
                    ImGui::ColorPicker3("Background Color", (float*)&localScene.backgroundColor);
                    ImGui::EndPopup();
                }
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);
            ImGui::Checkbox("Show Normals Instead of Color", &showNormals);
            ImGui::Checkbox("Show Wireframe", &showWireframe);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                auto quadsGenerator = quadwarp.getQuadsGenerator();
                ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator->params.correctOrientation);
                ImGui::DragFloat("Depth Threshold", &quadsGenerator->params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f");
                ImGui::DragFloat("Angle Threshold", &quadsGenerator->params.angleThreshold, 0.1f, 0.0f, 180.0f);
                ImGui::DragFloat("Flatten Threshold", &quadsGenerator->params.flattenThreshold, 0.001f, 0.0f, 1.0f);
                ImGui::DragFloat("Similarity Threshold", &quadsGenerator->params.proxySimilarityThreshold, 0.001f, 0.0f, 2.0f);
                ImGui::DragInt("Force Merge Iterations", &quadsGenerator->params.maxIterForceMerge, 0.1, 0, quadsGenerator->numQuadMaps/2);
            }

            ImGui::Separator();

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Send Reference Frame", ImVec2(buttonWidth, 0))) {
                sendReferenceFrame = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Send Residual Frame", ImVec2(buttonWidth, 0))) {
                sendResidualFrame = true;
            }
            ImGui::DragInt("Ref Frame Interval", &refFrameInterval, 0.1, 1, 5);

            ImGui::End();
        }

        if (showFramePreviewWindow) {
            flags = 0;
            ImGui::Begin("Reference Frame", 0, flags);
            ImGui::Image((void*)(intptr_t)(quadwarp.referenceFrameRT.colorTexture),
                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("Residual Frame (changed geometry)", 0, flags);
            ImGui::Image((void*)(intptr_t)(quadwarp.residualFrameMaskRT.colorTexture),
                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("Residual Frame (revealed geometry)", 0, flags);
            ImGui::Image((void*)(intptr_t)(quadwarp.residualFrameRT.colorTexture),
                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
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
    int frameCounter = 0;
    pose_id_t poseID = -1, prevPoseID = -1;
    app.onRender([&](double now, double dt) {
        // Handle keyboard input
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }
        totalDT += dt;

        if (rerenderIntervalMs > 0.0 && (now - lastRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendReferenceFrame = (frameCounter++) % refFrameInterval == 0; // insert Reference Frame every refFrameInterval frames
            sendResidualFrame = !sendReferenceFrame;
        }
        if (sendReferenceFrame || sendResidualFrame) {
            // Update all animations
            scene.updateAnimations(totalDT);
            totalDT = 0.0;
            lastRenderTime = now;

            poseID = poseReceiver.receivePose();
            if (poseID != -1 && poseID != prevPoseID) {
                // Offset camera
                camera.setPosition(camera.getPosition() + initialPosition);
                camera.updateViewMatrix();

                quadwarp.generateFrame(sendResidualFrame, showNormals, showDepth);

                // Restore camera position
                camera.setPosition(camera.getPosition() - initialPosition);
                camera.updateViewMatrix();

                spdlog::info("======================================================");
                spdlog::info("Rendering Time: {:.3f}ms", quadwarp.stats.totalRenderTime);
                spdlog::info("Create Proxies Time: {:.3f}ms", quadwarp.stats.totalCreateProxiesTime);
                spdlog::info("  Gen Quad Map Time: {:.3f}ms", quadwarp.stats.totalGenQuadMapTime);
                spdlog::info("  Simplify Time: {:.3f}ms", quadwarp.stats.totalSimplifyTime);
                spdlog::info("  Gather Quads Time: {:.3f}ms", quadwarp.stats.totalGatherQuadsTime);
                spdlog::info("Create Mesh Time: {:.3f}ms", quadwarp.stats.totaltimeToCreateMeshMs);
                spdlog::info("  Append Quads Time: {:.3f}ms", quadwarp.stats.totalAppendQuadsTime);
                spdlog::info("  Fill Output Quads Time: {:.3f}ms", quadwarp.stats.totalFillQuadsIndiciesTime);
                spdlog::info("  Create Vert/Ind Time: {:.3f}ms", quadwarp.stats.totalCreateVertIndTime);
                spdlog::info("Compress Time: {:.3f}ms", quadwarp.stats.totalCompressTime);
                if (showDepth) spdlog::info("Gen Depth Time: {:.3f}ms", quadwarp.stats.totalGenDepthTime);
                spdlog::info("Frame Size: {:.3f}MB", (quadwarp.stats.totalSizes.quadsSize +
                                                      quadwarp.stats.totalSizes.depthOffsetsSize) / BYTES_PER_MEGABYTE);
                spdlog::info("Num Proxies: {}Proxies", quadwarp.stats.totalSizes.numQuads);

                quadwarp.sendProxies(poseID, sendResidualFrame);

                sendReferenceFrame = false;
                sendResidualFrame = false;
                prevPoseID = poseID;
            }
        }

        // Show current mesh
        int currentIndex  = quadwarp.lastMeshIndex % 2;
        int previousIndex = (quadwarp.lastMeshIndex + 1) % 2;
        quadwarp.referenceFrameNodesLocal[currentIndex].visible = true;
        quadwarp.referenceFrameNodesLocal[previousIndex].visible = false;
        quadwarp.referenceFrameWireframesLocal[currentIndex].visible = true;
        quadwarp.referenceFrameWireframesLocal[currentIndex].visible = showWireframe;
        quadwarp.referenceFrameWireframesLocal[previousIndex].visible = false;
        quadwarp.residualFrameWireframesLocal.visible = quadwarp.residualFrameNodeLocal.visible && showWireframe;
        quadwarp.depthNode.visible = showDepth;

        // Offset camera
        camera.setPosition(camera.getPosition() + initialPosition);
        camera.updateViewMatrix();

        // Render generated meshes
        renderStats = renderer.drawObjects(localScene, camera);

        // Restore camera position
        camera.setPosition(camera.getPosition() - initialPosition);
        camera.updateViewMatrix();

        // Render to screen
        if (config.showWindow) {
            toneMapper.enableToneMapping(!showNormals);
            toneMapper.drawToScreen(renderer);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
