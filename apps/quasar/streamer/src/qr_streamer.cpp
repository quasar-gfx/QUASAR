#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>
#include <Renderers/DepthPeelingRenderer.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/SceneWindow.h>

#include <Streamers/QUASARStreamer.h>
#include <Receivers/PoseReceiver.h>
#include <HoleFiller.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "QUASAR Streamer";
    config.targetFramerate = 30;
    config.sortTransparent = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of local renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 80.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 140.0f);
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<int> targetBitrateIn(parser, "target-bitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 28);
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

    uint maxHiddenLayers = args::get(maxHiddenLayersIn);
    uint maxLayers = maxHiddenLayers + 2;

    uint targetBitrate = args::get(targetBitrateIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DepthPeelingRenderer remoteRendererDP(config, maxLayers - 1, true); // DP layers doesn't include wide fov
    DeferredRenderer remoteRenderer(config);

    // "Remote" scene
    Scene scene;
    PerspectiveCamera camera(remoteRendererDP.width, remoteRendererDP.height);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    float remoteFOV = args::get(remoteFOVIn);
    camera.setFovyDegrees(remoteFOV);

    glm::vec3 initialPosition = camera.getPosition();

    QuadSet quadSet(remoteWindowSize);
    float remoteFOVWide = args::get(remoteFOVWideIn);
    float viewSphereDiameter = args::get(viewSphereDiameterIn);
    QUASARStreamer quasar(
        quadSet,
        remoteRendererDP, remoteRenderer, scene, camera,
        {
            .maxLayers = maxLayers,
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteFOVWide,
            .targetBitRate = targetBitrate,
            .videoURL = videoURL,
            .proxiesURL = proxiesURL,
        });

    // "Local" scene for visualization
    Scene localScene;
    quasar.addMeshesToScene(localScene);

    PoseReceiver poseReceiver(&camera, poseURL);

    // Post processing
    HoleFiller holeFiller;

    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;

    bool sendReferenceFrame = true;
    bool sendResidualFrame = false;
    bool showResidualFrame = false;
    int refFrameInterval = 2;

    const double serverFPSValues[] = {0, 1, 2, 3, 4, 5};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "2 FPS", "3 FPS", "4 FPS", "5 FPS"};
    int serverFPSIndex = 1; // default to 1 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    bool* showLayers = new bool[maxLayers];
    for (int i = 0; i < maxLayers; i++) {
        showLayers[i] = true;
    }

    RenderStats renderStats;
    pose_id_t prevPoseID;
    FrameRateWindow frameRateWindow;
    SceneWindow sceneWindow(localScene, glm::vec2(430, 800));
    CameraHeader cameraHeader(camera, "Remote Camera", true);
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = true;
        static bool showFramePreviewWindow = false;
        static bool showLayerPreviews = false;

        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &frameRateWindow.visible);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Frame Previews", 0, &showFramePreviewWindow);
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Scene", 0, &sceneWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        sceneWindow.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            size_t totalTriangles = quasar.getNumTriangles();
            if (totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", totalTriangles);
            else if (totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               quasar.stats.proxySizes.numQuads,
                               quasar.stats.proxySizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quasar.stats.proxySizes.numDepthOffsets,
                               quasar.stats.proxySizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Proxies URL: %s", proxiesURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::Text("Client Pose ID: %d", prevPoseID);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                sendReferenceFrame = true;
            }
            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                sendReferenceFrame = true;
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                auto quadsGenerator = quasar.getQuadsGenerator();
                ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator->params.correctOrientation);
                ImGui::DragFloat("Depth Threshold", &quadsGenerator->params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f");
                ImGui::DragFloat("Angle Threshold", &quadsGenerator->params.angleThreshold, 0.1f, 0.0f, 180.0f);
                ImGui::DragFloat("Flatten Threshold", &quadsGenerator->params.flattenThreshold, 0.001f, 0.0f, 1.0f);
                ImGui::DragFloat("Plane Similarity Threshold", &quadsGenerator->params.planeSimilarityThreshold, 0.001f, 0.0f, 2.0f);
                ImGui::DragInt("Force Merge Iterations", &quadsGenerator->params.maxIterForceMerge, 1, 0, quadsGenerator->numQuadMaps);
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Video Stats")) {
                auto& videoStreamerRT = quasar.videoAtlasStreamerRT;
                ImGui::TextColored(ImVec4(1,0.5,0,1), "Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamerRT.getFrameRate(), 1000.0f / videoStreamerRT.getFrameRate());
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: %.3f ms", videoStreamerRT.stats.transferTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: %.3f ms", videoStreamerRT.stats.encodeTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: %.3f ms", videoStreamerRT.stats.sendTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0.5,1), "Bitrate: %.3f Mbps", videoStreamerRT.stats.bitrateMbps);
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

            ImGui::Separator();

            if (ImGui::DragFloat("View Sphere Diameter", &viewSphereDiameter, 0.025f, 0.1f, 2.0f)) {
                sendReferenceFrame = true;
                quasar.setViewSphereDiameter(viewSphereDiameter);
            }

            ImGui::Separator();

            const int columns = 3;
            for (int layer = 0; layer < maxLayers; layer++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(layer)).c_str(), &showLayers[layer]);
                if ((layer + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();
        }

        if (showFramePreviewWindow) {
            ImGui::Begin("Reference Frame", 0);
            ImGui::Image((void*)(intptr_t)(quasar.referenceFrameRT.colorTexture),
                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("Residual Frame (changed geometry)", 0);
            ImGui::Image((void*)(intptr_t)(quasar.residualFrameMaskRT.colorTexture),
                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("Residual Frame (revealed geometry)", 0);
            ImGui::Image((void*)(intptr_t)(quasar.residualFrameRT.colorTexture),
                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        if (showLayerPreviews) {
            for (int layer = 0; layer < maxLayers; layer++) {
                int viewIdx = maxLayers - layer - 1;
                if (showLayers[viewIdx]) {
                    ImGui::Begin(("View " + std::to_string(viewIdx)).c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
                    if (viewIdx == 0) {
                        ImGui::Image((void*)(intptr_t)(quasar.referenceFrameRT.colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    else {
                        ImGui::Image((void*)(intptr_t)(quasar.frameRTsHidLayer[viewIdx-1].colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    ImGui::End();
                }
            }
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        remoteRendererDP.setWindowSize(width, height);
        renderer.setWindowSize(width, height);
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });

    double totalDT = 0.0;
    double lastRenderTime = -INFINITY;
    int frameCounter = 0;
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

            pose_id_t poseID = poseReceiver.receivePose();
            if (poseID != -1 && poseID != prevPoseID) {
                // Offset camera
                camera.setPosition(camera.getPosition() + initialPosition);
                camera.updateViewMatrix();

                renderStats = quasar.generateFrame(sendResidualFrame, showNormals, showDepth);

                // Restore camera position
                camera.setPosition(camera.getPosition() - initialPosition);
                camera.updateViewMatrix();

                std::string frameType = sendReferenceFrame ? "Reference Frame" : "Residual Frame";
                spdlog::info("======================================================");
                spdlog::info("Rendering Time ({}): {:.3f}ms", frameType, quasar.stats.totalRenderTimeMs);
                spdlog::info("Create Proxies Time ({}): {:.3f}ms", frameType, quasar.stats.totalCreateProxiesTimeMs);
                spdlog::info("  Gen Quad Map Time ({}): {:.3f}ms", frameType, quasar.stats.totalGenQuadMapTimeMs);
                spdlog::info("  Simplify Time ({}): {:.3f}ms", frameType, quasar.stats.totalSimplifyTimeMs);
                spdlog::info("  Gather Quads Time ({}): {:.3f}ms", frameType, quasar.stats.totalGatherQuadsTime);
                spdlog::info("Create Mesh Time ({}): {:.3f}ms", frameType, quasar.stats.totalCreateMeshTimeMs);
                spdlog::info("  Append Quads Time ({}): {:.3f}ms", frameType, quasar.stats.totalAppendQuadsTimeMs);
                spdlog::info("  Create Vert/Ind Time ({}): {:.3f}ms", frameType, quasar.stats.totalCreateVertIndTimeMs);
                spdlog::info("Compress Time ({}): {:.3f}ms", frameType, quasar.stats.totalCompressTimeMs);
                if (showDepth) spdlog::info("Gen Depth Time ({}): {:.3f}ms", frameType, quasar.stats.totalGenDepthTimeMs);
                spdlog::info("Frame Size: {:.3f}MB", quasar.stats.frameSize / BYTES_PER_MEGABYTE);
                spdlog::info("Num Proxies: {}Proxies", quasar.stats.proxySizes.numQuads);

                prevPoseID = poseID;
                quasar.sendFrame(poseID, sendResidualFrame);

                showResidualFrame = sendResidualFrame;
                sendReferenceFrame = false;
                sendResidualFrame = false;
            }
        }

        int currentIndex  = quasar.lastMeshIndex % 2;
        int previousIndex = (quasar.lastMeshIndex + 1) % 2;
        for (int layer = 0; layer < maxLayers; layer++) {
            bool showLayer = showLayers[layer];
            if (layer == 0) {
                quasar.referenceFrameNodesLocal[currentIndex].visible = showLayer;
                quasar.referenceFrameNodesLocal[previousIndex].visible = false;
                quasar.referenceFrameWireframesLocal[currentIndex].visible = showLayer && showWireframe;
                quasar.referenceFrameWireframesLocal[previousIndex].visible = false;
                quasar.depthNode.visible = showLayer && showDepth;
            }
            else {
                quasar.nodesHidLayer[layer-1].visible = showLayer;
                quasar.wireframesHidLayer[layer-1].visible = showLayer && showWireframe;
                quasar.depthNodesHidLayer[layer-1].visible = showLayer && showDepth;
            }
        }
        quasar.residualFrameNodeLocal.visible = showResidualFrame;
        quasar.residualFrameWireframeLocal.visible = quasar.residualFrameNodeLocal.visible && showWireframe;

        // Offset camera
        camera.setPosition(camera.getPosition() + initialPosition);
        camera.updateViewMatrix();

        // Render generated meshes
        quasar.setDrawState(QuadMesh::DrawState::OPAQUE); // draw opaque quads first
        renderer.drawObjects(localScene, camera);
        quasar.setDrawState(QuadMesh::DrawState::TRANSPARENT); // then draw transparent quads
        renderer.drawObjects(localScene, camera, 0);

        // Restore camera position
        camera.setPosition(camera.getPosition() - initialPosition);
        camera.updateViewMatrix();

        // Render to screen
        if (config.showWindow) {
            auto quadsGenerator = quasar.getQuadsGenerator();
            holeFiller.enableTonemapping(!showNormals);
            holeFiller.setDepthThreshold(quadsGenerator->params.depthThreshold);
            holeFiller.drawToScreen(renderer);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
