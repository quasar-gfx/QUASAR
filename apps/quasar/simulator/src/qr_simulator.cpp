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
#include <UI/ScreenshotWindow.h>
#include <UI/RecordWindow.h>
#include <UI/TexturePreviewWindow.h>
#include <UI/SceneWindow.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <Streamers/QUASARStreamer.h>
#include <HoleFiller.h>

#include <NetworkSimulator.h>
#include <PosePredictor.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "QUASAR Simulator";
    config.sortTransparent = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> verbosity(parser, "verbosity", "Set log verbosity level", {'v', "verbosity"}, 2 /* spdlog::level::info */);
    args::ValueFlag<std::string> sizeIn(parser, "size", "Window resolution", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> rsizeIn(parser, "rsize", "Renderer resolution", {"rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<int> numPosesIn(parser, "num-poses", "Number of poses to load from camera path", {'n', "num-poses"}, -1);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::Flag posePredictionIn(parser, "pose-prediction", "Enable pose prediction", {'P', "pose-prediction"}, false);
    args::Flag poseSmoothingIn(parser, "pose-smoothing", "Enable pose smoothing", {'T', "pose-smoothing"}, false);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 80.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 140.0f);
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'l', "max-hidden-layers"}, 3);
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);
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

    // Parse arguments
    bool saveImagesToDisk = args::get(saveImages);
    int numPoses = args::get(numPosesIn);
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    // Parse render size
    std::string rsizeStr = args::get(rsizeIn);
    pos = rsizeStr.find('x');
    glm::uvec2 renderSize = glm::uvec2(std::stoi(rsizeStr.substr(0, pos)), std::stoi(rsizeStr.substr(pos + 1)));
    glm::uvec2 remoteWindowSize = renderSize;

    config.verbosity = args::get(verbosity);
    config.enableVSync = !args::get(novsync) && !saveImagesToDisk;
    config.showWindow = !saveImagesToDisk;
    uint maxHidLayers = args::get(maxHiddenLayersIn);
    uint maxLayers = maxHidLayers + 2;

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
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteRendererDP.width, remoteRendererDP.height);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);

    // "Local" scene
    Scene localScene;
    localScene.skybox = remoteScene.skybox;
    PerspectiveCamera camera(windowSize);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    QuadSet quadSet(remoteWindowSize);
    float remoteFOVWide = args::get(remoteFOVWideIn);
    float viewSphereDiameter = args::get(viewSphereDiameterIn);
    QUASARStreamer quasar(
        quadSet,
        remoteRendererDP, remoteRenderer, remoteScene, remoteCamera,
        {
            .maxLayers = static_cast<uint>(maxLayers),
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteFOVWide,
        });

    quasar.addMeshesToScene(localScene);

    // Post processing
    HoleFiller holeFiller;

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, holeFiller, outputPath, config.targetFramerate);

    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool hideReferenceFrame = false, hideResidualFrame = false;
    bool showResidualFrame = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewSphere = !cameraPathFileIn;

    bool sendReferenceFrame = true;
    bool sendResidualFrame = false;
    int refFrameInterval = 5;

    const double serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = posePredictionIn;
    bool poseSmoothing = poseSmoothingIn;
    NetworkSimulator networkSimulator({
        .networkLatencyMs = networkLatency,
        .networkJitterMs = networkJitter,
        .renderTimeMs = rerenderIntervalMs,
    });
    PosePredictor posePredictor({
        .enablePrediction = posePrediction,
        .enableSmoothing = poseSmoothing,
    });

    bool* showLayers = new bool[maxLayers];
    for (int i = 0; i < maxLayers; i++) {
        showLayers[i] = true;
    }

    FrameRateWindow frameRateWindow;
    ScreenshotWindow screenshotWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    TexturePreviewWindow videoPreviewWindow("Video Texture", quasar.videoAtlasStreamerRT.colorTexture, ImVec2(860, 860));
    TexturePreviewWindow alphaPreviewWindow("Alpha Texture", quasar.alphaAtlasRT.alphaTexture, ImVec2(860, 860));
    TexturePreviewWindow refFramePreviewWindow("Reference Frame", quasar.referenceFrameRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow resFrameChangedPreviewWindow("Residual Frame (changed geometry)", quasar.residualFrameMaskRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow resFrameFullPreviewWindow("Residual Frame (revealed geometry)", quasar.residualFrameRT.colorTexture, ImVec2(430, 270));
    SceneWindow sceneWindowRemote(remoteScene, ImVec2(430, 800));
    SceneWindow sceneWindowLocal(localScene, ImVec2(430, 800));
    CameraHeader cameraHeader(camera);

    CameraAnimator cameraAnimator(cameraPathFile, numPoses, !saveImagesToDisk); // Disable tweening when saving images
    if (saveImagesToDisk || cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        cameraAnimator.copyPoseToCamera(remoteCamera);
        spdlog::info("Loading camera path {} and saving images to {}", cameraPathFile.str(), outputPath.str());

        if (saveImagesToDisk) {
            recorder.setTargetFrameRate(-1 /* unlimited */);
            recorder.setFormat(Recorder::OutputFormat::PNG);
            recorder.start();
        }
    }

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImagesToDisk;
        static bool showMeshCapture = false;
        static bool showFramePreviewWindows = false;
        static bool showLayerPreviews = false;
        static bool saveAsSeparate = true;

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
            ImGui::MenuItem("Mesh Capture", 0, &showMeshCapture);
            ImGui::MenuItem("Frame Previews", 0, &showFramePreviewWindows);
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::MenuItem("Video Preview", 0, &videoPreviewWindow.visible);
            ImGui::MenuItem("Alpha Preview", 0, &alphaPreviewWindow.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Remote Scene", 0, &sceneWindowRemote.visible);
            ImGui::MenuItem("Local Scene", 0, &sceneWindowLocal.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Frame Capture")) {
            ImGui::MenuItem("Take Screenshot", 0, &screenshotWindow.visible);
            ImGui::MenuItem("Record Video", 0, &recordWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        screenshotWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        sceneWindowRemote.draw(now, dt);
        sceneWindowLocal.draw(now, dt);
        videoPreviewWindow.draw(now, dt);
        alphaPreviewWindow.draw(now, dt);

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

            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
            }
            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
            }
            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Hide Reference Frame", &hideReferenceFrame); ImGui::SameLine();
            ImGui::Checkbox("Hide Residual Frame", &hideResidualFrame);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                auto quadsGenerator = quasar.getQuadsGenerator();
                if (ImGui::Checkbox("Expand Proxies", &quadsGenerator->params.expandProxies)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator->params.correctOrientation)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Depth Threshold", &quadsGenerator->params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Angle Threshold", &quadsGenerator->params.angleThreshold, 0.1f, 0.0f, 180.0f)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Flatten Threshold", &quadsGenerator->params.flattenThreshold, 0.001f, 0.0f, 1.0f)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Plane Similarity Threshold", &quadsGenerator->params.planeSimilarityThreshold, 0.001f, 0.0f, 5.0f)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragInt("Force Merge Iterations", &quadsGenerator->params.maxIterForceMerge, 1, 0, quadsGenerator->numQuadMaps)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
            }

            ImGui::Separator();

            if (ImGui::DragFloat("Network Latency (ms)", &networkLatency, 0.5f, 0.0f, 500.0f)) {
                networkSimulator.setNetworkLatency(networkLatency);
            }
            if (ImGui::DragFloat("Network Jitter (ms)", &networkJitter, 0.25f, 0.0f, 50.0f)) {
                networkSimulator.setNetworkJitter(networkJitter);
            }

            ImGui::Checkbox("Pose Prediction Enabled", &posePredictor.enablePrediction);

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
                runAnimations = true;
            }

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Send Reference Frame", ImVec2(buttonWidth, 0))) {
                sendReferenceFrame = true;
                runAnimations = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Send Residual Frame", ImVec2(buttonWidth, 0))) {
                sendResidualFrame = true;
                runAnimations = true;
            }
            ImGui::DragInt("Ref Frame Interval", &refFrameInterval, 0.1, 1, 5);

            ImGui::Separator();

            if (ImGui::DragFloat("View Sphere Diameter", &viewSphereDiameter, 0.025f, 0.1f, 2.0f)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
                quasar.setViewSphereDiameter(viewSphereDiameter);
            }

            ImGui::Checkbox("Restrict Movement to View Sphere", &restrictMovementToViewSphere);

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

        if (showMeshCapture) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCapture);

            ImGui::Checkbox("Save as Separate Files", &saveAsSeparate);
            if (ImGui::Button("Save Proxies")) {
                if (!saveAsSeparate) {
                    std::vector<char> compressedData;
                    spdlog::info("Saved {} bytes to {}", quasar.writeToMemory(-1, sendResidualFrame, compressedData), outputPath.absolutePathStr());
                    Path filename = (outputPath / "frame").appendToName(".bin");
                    FileIO::writeToBinaryFile(filename, compressedData.data(), compressedData.size());
                    quasar.writeTexturesToFiles(outputPath);
                }
                else {
                    spdlog::info("Saved {} bytes to {}", quasar.writeToFiles(outputPath), outputPath.absolutePathStr());
                }
            }

            ImGui::End();
        }

        if (showFramePreviewWindows) {
            refFramePreviewWindow.visible = true; refFramePreviewWindow.draw(now, dt);
            resFrameChangedPreviewWindow.visible = true; resFrameChangedPreviewWindow.draw(now, dt);
            resFrameFullPreviewWindow.visible = true; resFrameFullPreviewWindow.draw(now, dt);
        }

        if (showLayerPreviews) {
            for (int layer = 0; layer < maxLayers; layer++) {
                int layerIdx = maxLayers - layer - 1;
                if (showLayers[layerIdx]) {
                    ImGui::Begin(("Layer " + std::to_string(layerIdx)).c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
                    if (layerIdx == 0) {
                        ImGui::Image((void*)(intptr_t)(quasar.referenceFrameRT.colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    else {
                        ImGui::Image((void*)(intptr_t)(quasar.frameRTsHidLayer[layerIdx-1].colorTexture.ID),
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
        // Handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = windowSize.x / 2.0;
            static float lastY = windowSize.y / 2.0;
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
        }
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (cameraAnimator.isRunning()) {
            bool waypointUpdated = cameraAnimator.update(saveImagesToDisk ? (1.0 / 60.0) : dt);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (waypointUpdated) {
                cameraAnimator.copyPoseToCamera(camera);
            }
        }
        else {
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
            camera.processKeyboard(keys, dt);
        }
        totalDT += dt;

        if (rerenderIntervalMs > 0.0 && (now - lastRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendReferenceFrame = (frameCounter++) % refFrameInterval == 0; // insert Reference Frame every refFrameInterval frames
            sendResidualFrame = !sendReferenceFrame;
        }
        if (sendReferenceFrame || sendResidualFrame) {
            // Update all animations
            if (runAnimations) {
                remoteScene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lastRenderTime = now;

            // "Send" pose to the server (simulates network latency)
            Pose currentPose;
            currentPose.setViewMatrix(camera.getViewMatrix());
            currentPose.setProjectionMatrix(camera.getProjectionMatrix());
            networkSimulator.sendPose(currentPose, now);
            posePredictor.addPose(currentPose);

            if (!preventCopyingLocalPose) {
                // "Receive" predicted pose to render (simulates network latency + prediction)
                Pose receivedPose;
                double originalTimestamp;
                if (networkSimulator.recvPose(receivedPose, now, originalTimestamp)) {
                    if (posePredictor.enablePrediction) {
                        Pose predictedPose;
                        double dtFuture = networkSimulator.getNetworkLatency() + networkSimulator.getRenderTime();
                        if (posePredictor.predictPose(predictedPose, now + dtFuture)) {
                            remoteCamera.setViewMatrix(predictedPose.mono.view);
                        }
                        else {
                            remoteCamera.setViewMatrix(receivedPose.mono.view);
                        }
                    }
                    else {
                        remoteCamera.setViewMatrix(receivedPose.mono.view);
                    }
                }
                // If we do not have a new pose, just send a new frame with the old pose
            }

            quasar.generateFrame(sendResidualFrame, showNormals, showDepth);
            quasar.sendFrame(-1, sendResidualFrame);

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

            showResidualFrame = sendResidualFrame;
            preventCopyingLocalPose = false;
            sendReferenceFrame = false;
            sendResidualFrame = false;
        }

        networkSimulator.update(now);

        // Hide/show nodes based on user input
        int currentIndex  = quasar.lastMeshIndex % 2;
        int previousIndex = (quasar.lastMeshIndex + 1) % 2;
        for (int layer = 0; layer < maxLayers; layer++) {
            bool showLayer = showLayers[layer];
            if (layer == 0) {
                quasar.referenceFrameNodesLocal[currentIndex].visible = showLayer && !hideReferenceFrame;
                quasar.referenceFrameNodesLocal[previousIndex].visible = false;
                quasar.referenceFrameWireframesLocal[currentIndex].visible = showLayer && !hideReferenceFrame && showWireframe;
                quasar.referenceFrameWireframesLocal[previousIndex].visible = false;
                quasar.depthNode.visible = showLayer && !hideReferenceFrame && showDepth;
            }
            else {
                quasar.nodesHidLayer[layer-1].visible = showLayer && !hideReferenceFrame;
                quasar.wireframesHidLayer[layer-1].visible = showLayer && !hideReferenceFrame && showWireframe;
                quasar.depthNodesHidLayer[layer-1].visible = showLayer && !hideReferenceFrame && showDepth;
            }
        }
        quasar.residualFrameNodeLocal.visible = showResidualFrame && !hideResidualFrame;
        quasar.residualFrameWireframeLocal.visible = quasar.residualFrameNodeLocal.visible && showWireframe;

        if (restrictMovementToViewSphere) {
            glm::vec3 remotePosition = remoteCamera.getPosition();
            glm::vec3 position = camera.getPosition();
            glm::vec3 direction = position - remotePosition;
            float distanceSquared = glm::dot(direction, direction);
            float radius = viewSphereDiameter / 2.0f;
            if (distanceSquared > radius * radius) {
                position = remotePosition + glm::normalize(direction) * radius;
            }
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        // Render generated meshes
        quasar.setDrawState(QuadMesh::DrawState::OPAQUE); // draw opaque quads first
        renderStats = renderer.drawObjects(localScene, camera);
        quasar.setDrawState(QuadMesh::DrawState::TRANSPARENT); // then draw transparent quads
        renderStats += renderer.drawObjects(localScene, camera, 0);

        // Render to screen
        auto quadsGenerator = quasar.getQuadsGenerator();
        holeFiller.enableTonemapping(!showNormals);
        holeFiller.setDepthThreshold(quadsGenerator->params.depthThreshold);
        holeFiller.drawToScreen(renderer);
        posePredictor.accumulateError(camera, remoteCamera);

        if (saveImagesToDisk) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.isRunning()) {
                auto errorStats = posePredictor.getErrorStats();
                spdlog::info("Pose Error:");
                spdlog::info("  Pos ({:.2f}±{:.2f},[{:.1f},{:.2f}])m",
                    errorStats.positionErrMeanStd.x, errorStats.positionErrMeanStd.y,
                    errorStats.positionErrMinMax.x, errorStats.positionErrMinMax.y);
                spdlog::info("  Rot ({:.2f}±{:.2f},[{:.1f},{:.2f}])°",
                    errorStats.rotationErrMeanStd.x, errorStats.rotationErrMeanStd.y,
                    errorStats.rotationErrMinMax.x, errorStats.rotationErrMinMax.y);
                spdlog::info("  RTT ({:.2f}±{:.2f})ms",
                    networkSimulator.getRTTMean(), networkSimulator.getRTTStdDev());
                recorder.stop();
                window->close();
            }
        }
        else if (recordWindow.isRecording()) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
