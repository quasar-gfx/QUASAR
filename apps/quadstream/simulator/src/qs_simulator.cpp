#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>
#include <PostProcessing/Tonemapper.h>

#include <UI/FrameRateWindow.h>
#include <UI/FrameCaptureWindow.h>
#include <UI/RecordWindow.h>
#include <UI/SceneWindow.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Streamers/QuadStreamStreamer.h>

#include <PoseSendRecvSimulator.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "QuadStream Simulator";
    config.sortTransparent = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::Flag posePredictionIn(parser, "pose-prediction", "Enable pose prediction", {'P', "pose-prediction"}, false);
    args::Flag poseSmoothingIn(parser, "pose-smoothing", "Enable pose smoothing", {'T', "pose-smoothing"}, false);
    args::ValueFlag<float> viewBoxSizeIn(parser, "view-box-size", "Size of view box in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<int> maxAdditionalViewsIn(parser, "views", "Max views", {'n', "max-views"}, 8);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 60.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 120.0f);
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

    config.enableVSync = !args::get(novsync) && !saveImages;
    config.showWindow = !args::get(saveImages);

    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();

    // 0th is standard view, maxViews-1 is large fov view
    int maxAdditionalViews = args::get(maxAdditionalViewsIn);
    int maxViews = maxAdditionalViews + 2;

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
    float remoteFOVWide = args::get(remoteFOVWideIn);
    float viewBoxSize = args::get(viewBoxSizeIn);
    QuadStreamStreamer quadstream(
        quadSet,
        remoteRenderer, remoteScene, remoteCamera,
        {
            .maxViews = static_cast<uint>(maxViews),
            .viewBoxSize = viewBoxSize,
            .wideFOV = remoteFOVWide,
        });

    quadstream.addMeshesToScene(localScene);

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
    CameraAnimator cameraAnimator(cameraPathFile);

    if (saveImages) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();
    }

    if (cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        cameraAnimator.copyPoseToCamera(remoteCamera);
    }

    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewBox = !cameraPathFileIn;

    bool sendRemoteFrame = true;

    const double serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = posePredictionIn;
    bool poseSmoothing = poseSmoothingIn;
    PoseSendRecvSimulator poseSendRecvSimulator({
        .networkLatencyMs = networkLatency,
        .networkJitterMs = networkJitter,
        .renderTimeMs = rerenderIntervalMs,
        .posePrediction = posePrediction,
        .poseSmoothing = poseSmoothing,
    });

    bool* showViews = new bool[maxViews];
    for (int view = 0; view < maxViews; ++view) {
        showViews[view] = true;
    }

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, glm::uvec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, glm::uvec2(430, 270), outputPath);
    SceneWindow sceneWindowRemote(remoteScene, glm::vec2(430, 800));
    SceneWindow sceneWindowLocal(localScene, glm::vec2(430, 800));
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImages;
        static bool showFramePreviews = false;
        static bool showMeshCapture = false;

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
            ImGui::MenuItem("Frame Capture", 0, &frameCaptureWindow.visible);
            ImGui::MenuItem("Record", 0, &recordWindow.visible);
            ImGui::MenuItem("Mesh Capture", 0, &showMeshCapture);
            ImGui::MenuItem("View Previews", 0, &showFramePreviews);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Remote Scene", 0, &sceneWindowRemote.visible);
            ImGui::MenuItem("Local Scene", 0, &sceneWindowLocal.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        frameCaptureWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        sceneWindowRemote.draw(now, dt);
        sceneWindowLocal.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            size_t totalTriangles = quadstream.getNumTriangles();
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
                               quadstream.stats.proxySizes.numQuads,
                               quadstream.stats.proxySizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quadstream.stats.proxySizes.numDepthOffsets,
                               quadstream.stats.proxySizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            const glm::vec3& position = camera.getPosition();
            if (ImGui::DragFloat3("Camera Position", (float*)&position, 0.01f)) {
                camera.setPosition(position);
            }
            const glm::vec3& rotation = camera.getRotationEuler();
            if (ImGui::DragFloat3("Camera Rotation", (float*)&rotation, 0.1f)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::DragFloat("Movement Speed", &camera.movementSpeed, 0.05f, 0.1f, 20.0f);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                sendRemoteFrame = true;
                runAnimations = false;
            }
            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                preventCopyingLocalPose = true;
                sendRemoteFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                auto quadsGenerator = quadstream.getQuadsGenerator();
                if (ImGui::Checkbox("Expand Proxies", &quadsGenerator->params.expandProxies)) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
                if (ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator->params.correctOrientation)) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Depth Threshold", &quadsGenerator->params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Angle Threshold", &quadsGenerator->params.angleThreshold, 0.1f, 0.0f, 180.0f)) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Flatten Threshold", &quadsGenerator->params.flattenThreshold, 0.001f, 0.0f, 1.0f)) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Similarity Threshold", &quadsGenerator->params.proxySimilarityThreshold, 0.001f, 0.0f, 2.0f)) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragInt("Force Merge Iterations", &quadsGenerator->params.maxIterForceMerge, 1, 0, quadsGenerator->numQuadMaps)) {
                    preventCopyingLocalPose = true;
                    sendRemoteFrame = true;
                    runAnimations = false;
                }
            }

            ImGui::Separator();

            if (ImGui::DragFloat("Network Latency (ms)", &networkLatency, 0.5f, 0.0f, 500.0f)) {
                poseSendRecvSimulator.setNetworkLatency(networkLatency);
            }
            if (ImGui::DragFloat("Network Jitter (ms)", &networkJitter, 0.25f, 0.0f, 50.0f)) {
                poseSendRecvSimulator.setNetworkJitter(networkJitter);
            }

            ImGui::Checkbox("Pose Prediction Enabled", &poseSendRecvSimulator.posePrediction);

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
                runAnimations = true;
            }

            if (ImGui::Button("Send Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                sendRemoteFrame = true;
                runAnimations = true;
            }

            ImGui::Separator();

            if (ImGui::DragFloat("View Box Size", &viewBoxSize, 0.025f, 0.1f, 2.0f)) {
                preventCopyingLocalPose = true;
                sendRemoteFrame = true;
                runAnimations = false;
                quadstream.setViewBoxSize(viewBoxSize);
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

            ImGui::Separator();

            const int columns = 4;
            for (int view = 0; view < maxViews; view++) {
                ImGui::Checkbox(("Show View " + std::to_string(view)).c_str(), &showViews[view]);
                if ((view + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();
        }

        if (showFramePreviews) {
            for (int view = 0; view < maxViews; view++) {
                int viewIdx = maxViews - view - 1;
                if (showViews[viewIdx]) {
                    ImGui::Begin(("View " + std::to_string(viewIdx)).c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
                    ImGui::Image((void*)(intptr_t)(quadstream.referenceFrameRTs[viewIdx].colorTexture.ID),
                                 ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }
            }
        }

        if (showMeshCapture) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCapture);

            if (ImGui::Button("Save Proxies")) {
                spdlog::info("Saved {} bytes to {}", quadstream.writeToFiles(outputPath), outputPath.absolutePathStr());
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
    bool updateClient = !saveImages;
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

        if (cameraAnimator.running) {
            updateClient = cameraAnimator.update(!cameraPathFileIn ? dt : 1.0 / MILLISECONDS_IN_SECOND);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (updateClient) {
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
            sendRemoteFrame = true;
        }
        if (sendRemoteFrame) {
            // Update all animations
            if (runAnimations) {
                remoteScene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lastRenderTime = now;

            // "Send" pose to the server. this will wait until latency+/-jitter ms have passed
            poseSendRecvSimulator.sendPose(camera, now);
            if (!preventCopyingLocalPose) {
                // "Receive" a predicted pose to render a new frame. this will wait until latency+/-jitter ms have passed
                Pose clientPosePred;
                if (poseSendRecvSimulator.recvPoseToRender(clientPosePred, now)) {
                    remoteCamera.setViewMatrix(clientPosePred.mono.view);
                }
                // If we do not have a new pose, just send a new frame with the old pose
            }

            quadstream.generateFrame(showNormals, showDepth);

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.3f}ms", quadstream.stats.totalRenderTimeMs);
            spdlog::info("Create Proxies Time: {:.3f}ms", quadstream.stats.totalCreateProxiesTimeMs);
            spdlog::info("  Gen Quad Map Time: {:.3f}ms", quadstream.stats.totalGenQuadMapTimeMs);
            spdlog::info("  Simplify Time: {:.3f}ms", quadstream.stats.totalSimplifyTimeMs);
            spdlog::info("  Gather Quads Time: {:.3f}ms", quadstream.stats.totalGatherQuadsTime);
            spdlog::info("Create Mesh Time: {:.3f}ms", quadstream.stats.totalCreateMeshTimeMs);
            spdlog::info("  Append Quads Time: {:.3f}ms", quadstream.stats.totalAppendQuadsTimeMs);
            spdlog::info("  Create Vert/Ind Time: {:.3f}ms", quadstream.stats.totalCreateVertIndTimeMs);
            spdlog::info("Compress Time: {:.3f}ms", quadstream.stats.totalCompressTimeMs);
            if (showDepth) spdlog::info("Gen Depth Time: {:.3f}ms", quadstream.stats.totalGenDepthTimeMs);
            spdlog::info("Frame Size: {:.3f}MB", quadstream.stats.frameSize / BYTES_PER_MEGABYTE);
            spdlog::info("Num Proxies: {}Proxies", quadstream.stats.proxySizes.numQuads);

            preventCopyingLocalPose = false;
            sendRemoteFrame = false;
        }

        poseSendRecvSimulator.update(now);

        // Hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showView = showViews[view];
            quadstream.referenceFrameNodesLocal[view].visible = showView;
            quadstream.referenceFrameWireframesLocal[view].visible = showView && showWireframe;
            quadstream.depthNodes[view].visible = showView && showDepth;
        }

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

        double startTime = window->getTime();

        // Render all objects in scene
        renderStats = renderer.drawObjects(localScene, camera);

        // Render to screen
        tonemapper.enableTonemapping(!showNormals);
        tonemapper.drawToScreen(renderer);
        if (!updateClient) {
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCamera);

        if (cameraPathFileIn) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.running) {
                poseSendRecvSimulator.printErrors();
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
