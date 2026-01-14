#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DepthPeelingRenderer.h> // We use depth peeling here to be consistent with other baselines
#include <PostProcessing/Tonemapper.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/FrameCaptureWindow.h>
#include <UI/RecordWindow.h>
#include <UI/TexturePreviewWindow.h>
#include <UI/SceneWindow.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Streamers/MeshWarpStreamer.h>
#include <NetworkSimulator.h>
#include <PosePredictor.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
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
    args::ValueFlag<uint> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<uint> vertexGroupSizeIn(parser, "vertex", "Size of vertex grouping", {'g', "vertex-group-size"}, 1);
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

    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);
    int numPoses = args::get(numPosesIn);

    uint depthFactor = args::get(depthFactorIn);
    uint vertexGroupSize = args::get(vertexGroupSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DepthPeelingRenderer remoteRenderer(config);

    // "Remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);

    // "Local" scene
    Scene scene;
    scene.skybox = remoteScene.skybox;
    PerspectiveCamera camera(windowSize);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    MeshWarpStreamer meshWarpStreamer(
        remoteRenderer, remoteScene, remoteCamera,
        {
            .depthFactor = depthFactor,
            .vertexGroupSize = vertexGroupSize,
        });

    Node node(&meshWarpStreamer.getMesh());
    node.frustumCulled = false;
    scene.addChildNode(&node);

    UnlitMaterial wireframeMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    Node nodeWireframe(&meshWarpStreamer.getMesh());
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = &wireframeMaterial;
    scene.addChildNode(&nodeWireframe);

    UnlitMaterial pointCloudMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) });
    Node nodePointCloud(&meshWarpStreamer.getMesh());
    nodePointCloud.frustumCulled = false;
    nodePointCloud.primitiveType = GL_POINTS;
    nodePointCloud.pointSize = 7.5f;
    nodePointCloud.visible = false;
    nodePointCloud.overrideMaterial = &pointCloudMaterial;
    scene.addChildNode(&nodePointCloud);

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
    CameraAnimator cameraAnimator(cameraPathFile, numPoses);

    if (saveImages) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();
    }

    if (cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        cameraAnimator.copyPoseToCamera(remoteCamera);
    }

    bool showWireframe = false;
    bool showDepth = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;

    bool sendRemoteFrame = true;

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

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    TexturePreviewWindow videoPreviewWindow("Video Texture", meshWarpStreamer.renderTarget.colorTexture, ImVec2(430, 270));
    SceneWindow sceneWindow(scene, ImVec2(430, 800));
    CameraHeader cameraHeader(camera);
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImages;
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
            ImGui::MenuItem("Frame Preview", 0, &videoPreviewWindow.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Scene", 0, &sceneWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        frameCaptureWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        sceneWindow.draw(now, dt);
        videoPreviewWindow.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Data Size: %.3f MB", static_cast<float>(meshWarpStreamer.stats.compressedSize) / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::DragFloat("Remote FOV", &remoteFOV, 0.1f, 80.0f, 180.0f)) {
                remoteCamera.setFovyDegrees(remoteFOV);

                preventCopyingLocalPose = true;
                sendRemoteFrame = true;
                runAnimations = false;
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

            if (ImGui::Button("Send Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                sendRemoteFrame = true;
                runAnimations = true;
            }

            ImGui::End();
        }

        if (showMeshCapture) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCapture);

            if (ImGui::Button("Save Depth")) {
                spdlog::info("Saved {} bytes to {}", meshWarpStreamer.writeToFiles(outputPath), outputPath.absolutePathStr());
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
    double lasttotalRenderTime = -INFINITY;
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

        if (rerenderIntervalMs > 0.0 && (now - lasttotalRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendRemoteFrame = true;
        }
        if (sendRemoteFrame) {
            // Update all animations
            if (runAnimations) {
                remoteScene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lasttotalRenderTime = now;

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

            // Generate new frame
            meshWarpStreamer.generateFrame();

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.3f}ms", meshWarpStreamer.stats.totalRenderTimeMs);
            spdlog::info("Create Mesh Time: {:.3f}ms", meshWarpStreamer.stats.totalGenMeshTime);
            spdlog::info("Compress Time: {:.3f}ms", meshWarpStreamer.stats.totalCompressTimeMs);
            spdlog::info("Frame Size: {:.3f}MB", static_cast<float>(meshWarpStreamer.stats.compressedSize) / BYTES_PER_MEGABYTE);

            preventCopyingLocalPose = false;
            sendRemoteFrame = false;
        }

        networkSimulator.update(now);

        nodeWireframe.visible = showWireframe;
        nodePointCloud.visible = showDepth;

        double startTime = window->getTime();

        // Render generated meshes
        renderStats = renderer.drawObjects(scene, camera);

        tonemapper.drawToScreen(renderer);
        if (!updateClient) {
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        posePredictor.accumulateError(camera, remoteCamera);

        if (cameraPathFileIn) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.running) {
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
