#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <SceneLoader.h>

#include <Renderers/DeferredRenderer.h>
#include <Renderers/ForwardRenderer.h>

#include <CameraAnimator.h>
#include <Recorder.h>

#include <PostProcessing/Tonemapper.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/RecordWindow.h>
#include <UI/SceneWindow.h>
#include <UI/ScreenshotWindow.h>
#include <UI/TexturePreviewWindow.h>

#include <NetworkSimulator.h>
#include <PosePredictor.h>
#include <Streamers/QuadsStreamer.h>
#include <Utils/TimeUtils.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Simulator";
    config.sortTransparent = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> verbosity(parser, "verbosity", "Set log verbosity level", {'v', "verbosity"}, SPDLOG_LEVEL_INFO);
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<std::string> sizeIn(parser, "size", "Window resolution", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> rsizeIn(parser, "rsize", "Renderer resolution", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<int> numPosesIn(parser, "num-poses", "Number of poses to load from camera path", {'n', "num-poses"}, -1);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::Flag posePredictionIn(parser, "pose-prediction", "Enable pose prediction", {'P', "pose-prediction"}, false);
    args::Flag poseSmoothingIn(parser, "pose-smoothing", "Enable pose smoothing", {'T', "pose-smoothing"}, false);
    args::ValueFlag<float> viewBoxSizeIn(parser, "view-box-size", "Size of view box in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 80.0f);
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
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteRenderer.width, remoteRenderer.height);
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
    QuadsStreamer quadwarp(quadSet, remoteRenderer, remoteScene, remoteCamera);

    // Add meshes to local scene
    quadwarp.addMeshesToScene(localScene);

    // Post processing
    Tonemapper tonemapper;

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
    }, renderer, tonemapper, outputPath, config.targetFramerate);

    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool hideReferenceFrame = false, hideResidualFrame = false;
    bool showResidualFrame = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewBox = !cameraPathFileIn;
    float viewBoxSize = args::get(viewBoxSizeIn);

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

    FrameRateWindow frameRateWindow;
    ScreenshotWindow screenshotWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    TexturePreviewWindow videoPreviewWindow("Video Texture", quadwarp.videoAtlasStreamerRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow alphaPreviewWindow("Alpha Texture", quadwarp.alphaAtlasRT.alphaTexture, ImVec2(430, 270));
    TexturePreviewWindow refFramePreviewWindow("Reference Frame", quadwarp.referenceFrameRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow resFrameChangedPreviewWindow("Residual Frame (changed geometry)", quadwarp.residualFrameMaskRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow resFrameFullPreviewWindow("Residual Frame (revealed geometry)", quadwarp.residualFrameRT.colorTexture, ImVec2(430, 270));
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

            size_t totalTriangles = quadwarp.getNumTriangles();
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
                               quadwarp.stats.proxySizes.numQuads,
                               quadwarp.stats.proxySizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quadwarp.stats.proxySizes.numDepthOffsets,
                               quadwarp.stats.proxySizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

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
                auto quadsGenerator = quadwarp.getQuadsGenerator();
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

            if (ImGui::DragFloat("View Box Size", &viewBoxSize, 0.025f, 0.1f, 2.0f)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

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
                    spdlog::info("Saved {} bytes to {}", quadwarp.writeToMemory(-1, sendResidualFrame, compressedData), outputPath.absolutePathStr());
                    Path filename = (outputPath / "frame").appendToName(".bin");
                    FileIO::writeToBinaryFile(filename, compressedData.data(), compressedData.size());
                    quadwarp.writeTexturesToFiles(outputPath);
                }
                else {
                    spdlog::info("Saved {} bytes to {}", quadwarp.writeToFiles(outputPath), outputPath.absolutePathStr());
                }
            }

            ImGui::End();
        }

        if (showFramePreviewWindows) {
            refFramePreviewWindow.visible = true; refFramePreviewWindow.draw(now, dt);
            resFrameChangedPreviewWindow.visible = true; resFrameChangedPreviewWindow.draw(now, dt);
            resFrameFullPreviewWindow.visible = true; resFrameFullPreviewWindow.draw(now, dt);
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
            currentPose.timestamp = static_cast<double>(timeutils::secondsToMicros(now));
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

            renderStats = quadwarp.generateFrame(sendResidualFrame, showNormals, showDepth);
            quadwarp.sendFrame(-1, sendResidualFrame);

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.3f}ms", quadwarp.stats.totalRenderTimeMs);
            spdlog::info("Create Proxies Time: {:.3f}ms", quadwarp.stats.totalCreateProxiesTimeMs);
            spdlog::info("  Gen Quad Map Time: {:.3f}ms", quadwarp.stats.totalGenQuadMapTimeMs);
            spdlog::info("  Simplify Time: {:.3f}ms", quadwarp.stats.totalSimplifyTimeMs);
            spdlog::info("  Gather Quads Time: {:.3f}ms", quadwarp.stats.totalGatherQuadsTime);
            spdlog::info("Create Mesh Time: {:.3f}ms", quadwarp.stats.totalCreateMeshTimeMs);
            spdlog::info("  Append Quads Time: {:.3f}ms", quadwarp.stats.totalAppendQuadsTimeMs);
            spdlog::info("  Create Vert/Ind Time: {:.3f}ms", quadwarp.stats.totalCreateVertIndTimeMs);
            spdlog::info("Compress Time: {:.3f}ms", quadwarp.stats.totalCompressTimeMs);
            if (showDepth) spdlog::info("Gen Depth Time: {:.3f}ms", quadwarp.stats.totalGenDepthTimeMs);
            spdlog::info("Frame Size: {:.3f}MB", quadwarp.stats.frameSize / BYTES_PER_MEGABYTE);
            spdlog::info("Num Proxies: {}Proxies", quadwarp.stats.proxySizes.numQuads);

            showResidualFrame = sendResidualFrame;
            preventCopyingLocalPose = false;
            sendReferenceFrame = false;
            sendResidualFrame = false;
        }

        networkSimulator.update(now);

        // Show meshes
        int currentIndex  = quadwarp.lastMeshIndex % 2;
        int previousIndex = (quadwarp.lastMeshIndex + 1) % 2;
        quadwarp.referenceFrameNodesLocal[currentIndex].visible = !hideReferenceFrame;
        quadwarp.referenceFrameNodesLocal[previousIndex].visible = false;
        quadwarp.referenceFrameWireframesLocal[currentIndex].visible = !hideReferenceFrame && showWireframe;
        quadwarp.referenceFrameWireframesLocal[previousIndex].visible = false;
        quadwarp.residualFrameNodeLocal.visible = showResidualFrame && !hideResidualFrame;
        quadwarp.residualFrameWireframeLocal.visible = quadwarp.residualFrameNodeLocal.visible && showWireframe;
        quadwarp.depthNode.visible = !hideReferenceFrame && showDepth;

        if (restrictMovementToViewBox) {
            glm::vec3 remotePosition = remoteCamera.getPosition();
            glm::vec3 position = camera.getPosition();
            // Restrict camera position to be inside position±viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - viewBoxSize/2, remotePosition.x + viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewBoxSize/2, remotePosition.y + viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewBoxSize/2, remotePosition.z + viewBoxSize/2);
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        // Render generated meshes
        renderStats = renderer.drawObjects(localScene, camera);

        // Render to screen
        tonemapper.enableTonemapping(!showNormals);
        tonemapper.drawToScreen(renderer);
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
