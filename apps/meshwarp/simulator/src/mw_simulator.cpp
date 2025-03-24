#include <filesystem>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowDepthEffect.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <BC4DepthStreamer.h>
#include <PoseSendRecvSimulator.h>

#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "3840x2160");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::ValueFlag<unsigned int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
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

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    }

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    // parse remote size
    std::string rsizeStr = args::get(resIn);
    pos = rsizeStr.find('x');
    glm::uvec2 remoteWindowSize = glm::uvec2(std::stoi(rsizeStr.substr(0, pos)), std::stoi(rsizeStr.substr(pos + 1)));

    config.enableVSync = !args::get(novsync) && !saveImages;
    config.showWindow = !args::get(saveImages);

    std::string sceneFile = args::get(sceneFileIn);
    std::string cameraPathFile = args::get(cameraPathFileIn);
    std::string dataPath = args::get(dataPathIn);
    if (dataPath.back() != '/') {
        dataPath += "/";
    }
    // create data path if it doesn't exist
    if (!std::filesystem::exists(dataPath)) {
        std::filesystem::create_directories(dataPath);
    }

    unsigned int surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DeferredRenderer remoteRenderer(config);

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float fov = args::get(fovIn);
    remoteCamera.setFovyDegrees(fov);

    // "local" scene
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    RenderTarget renderTarget({
        .width = remoteWindowSize.x,
        .height = remoteWindowSize.y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    BC4DepthStreamer bc4DepthStreamerRT = BC4DepthStreamer({
        .width = remoteWindowSize.x,
        .height = remoteWindowSize.y,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    glm::uvec2 depthMapSize = remoteWindowSize;
    glm::uvec2 adjustedWindowSize = depthMapSize / surfelSize;

    unsigned int maxVertices = adjustedWindowSize.x * adjustedWindowSize.y;
    unsigned int numTriangles = (adjustedWindowSize.x-1) * (adjustedWindowSize.y-1) * 2;
    unsigned int maxIndices = numTriangles * 3;

    Mesh mesh = Mesh({
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .material = new UnlitMaterial({ .baseColorTexture = &renderTarget.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Node nodeWireframe = Node(&mesh);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    scene.addChildNode(&nodeWireframe);

    Node nodePointCloud = Node(&mesh);
    nodePointCloud.frustumCulled = false;
    nodePointCloud.primativeType = GL_POINTS;
    nodePointCloud.pointSize = 7.5f;
    nodePointCloud.visible = false;
    nodePointCloud.overrideMaterial = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) });
    scene.addChildNode(&nodePointCloud);

    // shaders
    ComputeShader meshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_MESHFROMBC4_COMP,
        .computeCodeSize = SHADER_COMMON_MESHFROMBC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    // post processing
    ToneMapper toneMapper;
    ShowDepthEffect showDepthEffect(camera);

    Recorder recorder(renderer, toneMapper, dataPath, config.targetFramerate);
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

    bool showWireframe = false;
    bool showDepth = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;

    bool renderRemoteFrame = true;

    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps
    double rerenderInterval = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = true;
    PoseSendRecvSimulator poseSendRecvSimulator(networkLatency, networkJitter, rerenderInterval / MILLISECONDS_IN_SECOND);

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = !saveImages;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps

        static bool showSkyBox = true;

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

            glm::vec3 position = camera.getPosition();
            if (ImGui::DragFloat3("Camera Position", (float*)&position, 0.01f)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::DragFloat3("Camera Rotation", (float*)&rotation, 0.1f)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::DragFloat("Movement Speed", &camera.movementSpeed, 0.05f, 0.1f, 20.0f);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Background Settings")) {
                if (ImGui::Checkbox("Show Sky Box", &showSkyBox)) {
                    scene.envCubeMap = showSkyBox ? remoteScene.envCubeMap : nullptr;
                }

                if (ImGui::Button("Change Background Color", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    ImGui::OpenPopup("Background Color Popup");
                }
                if (ImGui::BeginPopup("Background Color Popup")) {
                    ImGui::ColorPicker3("Background Color", (float*)&scene.backgroundColor);
                    ImGui::EndPopup();
                }
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::DragFloat("FoV (degrees)", &fov, 0.1f, 60.0f, 120.0f)) {
                remoteCamera.setFovyDegrees(fov);
                remoteCamera.updateProjectionMatrix();

                preventCopyingLocalPose = true;
                renderRemoteFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::DragFloat("Network Latency (ms)", &networkLatency, 0.5f, 0.0f, 500.0f)) {
                poseSendRecvSimulator.setNetworkLatency(networkLatency);
            }

            if (ImGui::DragFloat("Network Jitter (ms)", &networkJitter, 0.25f, 0.0f, 50.0f)) {
                poseSendRecvSimulator.setNetworkJitter(networkJitter);
            }

            if (ImGui::Checkbox("Pose Prediction Enabled", &posePrediction)) {
                poseSendRecvSimulator.setPosePrediction(posePrediction);
            }

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderInterval = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            if (ImGui::Button("Send Server Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                renderRemoteFrame = true;
                runAnimations = true;
            }

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = dataPath + std::string(fileNameBase) + "." +
                                              std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    double lastRenderTime = 0.0;
    bool updateClient = !saveImages;
    app.onRender([&](double now, double dt) {
        // handle mouse input
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

        // update all animations
        // we run this outside the remote loop to ensure that animations are sync'ed with scene_viewer
        if (runAnimations) {
            remoteScene.updateAnimations(dt);
        }

        if (rerenderInterval > 0.0 && (now - lastRenderTime) >= (rerenderInterval - 1) / MILLISECONDS_IN_SECOND) {
            renderRemoteFrame = true;
            lastRenderTime = now;
        }
        if (renderRemoteFrame) {
            double startTime = window->getTime();
            double totalRenderTime = 0.0;
            double totalGenMeshTime = 0.0;
            double totalCreateVertIndTime = 0.0;
            double totalCompressTime = 0.0;

            unsigned int compressedSize = 0;

            // "send" pose to the server. this will wait until latency+/-jitter ms have passed
            poseSendRecvSimulator.sendPose(camera, now);
            if (!preventCopyingLocalPose) {
                // "receive" a predicted pose to render a new frame. this will wait until latency+/-jitter ms have passed
                Pose clientPosePred;
                if (poseSendRecvSimulator.recvPoseToRender(clientPosePred, now)) {
                    remoteCamera.setViewMatrix(clientPosePred.mono.view);
                }
                // if we do not have a new pose, just send a new frame with the old pose
            }

            // render remoteScene
            remoteRenderer.drawObjects(remoteScene, remoteCamera);

            // copy rendered result to video render target
            toneMapper.enableToneMapping(false);
            toneMapper.drawToRenderTarget(remoteRenderer, renderTarget);
            showDepthEffect.drawToRenderTarget(remoteRenderer, bc4DepthStreamerRT);

            totalRenderTime += timeutils::secondsToMillis(window->getTime() - startTime);

            startTime = window->getTime();
            compressedSize = bc4DepthStreamerRT.compress(true);
            totalCompressTime += timeutils::secondsToMillis(window->getTime() - startTime);

            startTime = window->getTime();
            meshFromBC4Shader.startTiming();
            meshFromBC4Shader.bind();
            {
                meshFromBC4Shader.setBool("unlinearizeDepth", true);
                meshFromBC4Shader.setVec2("depthMapSize", remoteWindowSize);
                meshFromBC4Shader.setInt("surfelSize", surfelSize);
            }
            {
                meshFromBC4Shader.setMat4("projection", remoteCamera.getProjectionMatrix());
                meshFromBC4Shader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
                meshFromBC4Shader.setMat4("viewColor", remoteCamera.getViewMatrix());
                meshFromBC4Shader.setMat4("viewInverseDepth", remoteCamera.getViewMatrixInverse());

                meshFromBC4Shader.setFloat("near", remoteCamera.getNear());
                meshFromBC4Shader.setFloat("far", remoteCamera.getFar());
            }
            {
                meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
                meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
                meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, bc4DepthStreamerRT.bc4CompressedBuffer);
            }
            // dispatch compute shader to generate vertices and indices for mesh
            meshFromBC4Shader.dispatch((adjustedWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (adjustedWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
            meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                            GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
            meshFromBC4Shader.endTiming();
            totalGenMeshTime += timeutils::secondsToMillis(window->getTime() - startTime);
            totalCreateVertIndTime += meshFromBC4Shader.getElapsedTime();

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.2f}ms", totalRenderTime);
            spdlog::info("Create Mesh Time: {:.2f}ms", totalGenMeshTime);
            spdlog::info("  Create Vert/Ind Time: {:.2f}ms", totalCreateVertIndTime);
            spdlog::info("Compress Time: {:.3f}ms", totalCompressTime);
            spdlog::info("Frame Size: {:.3f}MB", static_cast<float>(compressedSize) / BYTES_IN_MB);

            preventCopyingLocalPose = false;
            renderRemoteFrame = false;
        }

        poseSendRecvSimulator.update(now);

        if (!updateClient) {
            return;
        }

        nodeWireframe.visible = showWireframe;
        nodePointCloud.visible = showDepth;

        double startTime = window->getTime();

        // render generated meshes
        renderStats = renderer.drawObjects(scene, camera);

        toneMapper.enableToneMapping(true);
        toneMapper.drawToScreen(renderer);
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCamera);

        if ((cameraPathFileIn && cameraAnimator.running) || recording) {
            recorder.captureFrame(camera);
        }
        if (cameraPathFileIn && !cameraAnimator.running) {
            poseSendRecvSimulator.printErrors();

            recorder.captureFrame(camera); // capture final frame
            recorder.stop();
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
