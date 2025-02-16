#include <filesystem>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <PostProcessing/ToneMapper.h>

#include <Recorder.h>
#include <Animator.h>

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
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'V', "vsync"}, true);
    args::Flag saveImage(parser, "save", "Save outputs to disk", {'I', "save-image"});
    args::ValueFlag<std::string> animationFileIn(parser, "anim-path", "Path to camera animation file", {'A', "animation-path"});
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

    config.enableVSync = args::get(vsyncIn);
    config.showWindow = !args::get(saveImage);

    std::string sceneFile = args::get(sceneFileIn);
    std::string animationFile = args::get(animationFileIn);
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
    ForwardRenderer remoteRenderer(config);
    ForwardRenderer renderer(config);

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float fov = args::get(fovIn);
    remoteCamera.setFovyDegrees(fov);

    // scene with all the meshes
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    RenderTarget renderTarget({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    glm::uvec2 depthMapSize = windowSize;
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
    ComputeShader meshFromDepthShader({
        .computeCodeData = SHADER_COMMON_MESHFROMDEPTH_COMP,
        .computeCodeSize = SHADER_COMMON_MESHFROMDEPTH_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    // post processing
    ToneMapper toneMapper;

    Recorder recorder(renderer, toneMapper, dataPath, config.targetFramerate);
    Animator animator(animationFile);

    if (saveImage) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();

        animator.copyPoseToCamera(camera);
        animator.copyPoseToCamera(remoteCamera);
    }

    bool showWireframe = false;
    bool showDepth = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = animationFileIn;

    bool rerender = true;

    float networkLatency = !animationFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !animationFileIn ? 0.0f : args::get(networkJitterIn);
    PoseSendRecvSimulator poseSendRecvSimulator(networkLatency, networkJitter);
    bool posePrediction = true;
    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !animationFileIn ? 0 : 5; // default to 30fps
    double rerenderInterval = MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !animationFileIn ? 0 : 5; // default to 30fps

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
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 0.1f, 20.0f);

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

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::SliderFloat("FoV (degrees)", &fov, 60.0f, 120.0f)) {
                remoteCamera.setFovyDegrees(fov);
                remoteCamera.updateProjectionMatrix();

                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::SliderFloat("Network Latency (ms)", &networkLatency, 0.0f, 500.0f)) {
                poseSendRecvSimulator.setNetworkLatency(networkLatency);
            }

            if (ImGui::SliderFloat("Network Jitter (ms)", &networkJitter, 0.0f, 50.0f)) {
                poseSendRecvSimulator.setNetworkJitter(networkJitter);
            }

            if (ImGui::Checkbox("Pose Prediction Enabled", &posePrediction)) {
                poseSendRecvSimulator.setPosePrediction(posePrediction);
            }

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderInterval = MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            if (ImGui::Button("Send Server Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                rerender = true;
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

        if (animator.running) {
            animator.copyPoseToCamera(camera);
            animator.update(dt);
        }
        else {
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
            camera.processKeyboard(keys, dt);
        }

        if (animationFileIn) {
            now = animator.now;
            dt = animator.dt;
        }

        // update all animations
        if (runAnimations) {
            remoteScene.updateAnimations(dt);
        }

        if (rerenderInterval > 0.0 && now - lastRenderTime >= rerenderInterval / MILLISECONDS_IN_SECOND) {
            rerender = true;
            runAnimations = true;
            lastRenderTime = now;
        }
        if (rerender) {
            double startTime = window->getTime();
            double totalRenderTime = 0.0;
            double totalGenMeshTime = 0.0;
            double totalCreateVertIndTime = 0.0;

            poseSendRecvSimulator.sendPose(camera, now);
            if (!preventCopyingLocalPose) {
                Pose clientPose;
                if (poseSendRecvSimulator.recvPose(clientPose, now)) {
                    remoteCamera.setViewMatrix(clientPose.mono.view);
                    poseSendRecvSimulator.accumulateError(camera, remoteCamera);
                }
            }

            // render remoteScene
            remoteRenderer.drawObjects(remoteScene, remoteCamera);

            // copy rendered result to video render target
            toneMapper.enableToneMapping(false);
            toneMapper.drawToRenderTarget(remoteRenderer, renderTarget);

            totalRenderTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;

            startTime = window->getTime();
            meshFromDepthShader.startTiming();

            meshFromDepthShader.bind();
            {
                meshFromDepthShader.setTexture(remoteRenderer.gBuffer.depthStencilBuffer, 0);
            }
            {
                meshFromDepthShader.setVec2("depthMapSize", depthMapSize);
                meshFromDepthShader.setInt("surfelSize", surfelSize);
            }
            {
                meshFromDepthShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                meshFromDepthShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
                meshFromDepthShader.setMat4("view", remoteCamera.getViewMatrix());
                meshFromDepthShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());

                meshFromDepthShader.setFloat("near", remoteCamera.getNear());
                meshFromDepthShader.setFloat("far", remoteCamera.getFar());
            }
            {
                meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
                meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
            }
            meshFromDepthShader.dispatch((adjustedWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                         (adjustedWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
            meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

            meshFromDepthShader.endTiming();
            totalGenMeshTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
            totalCreateVertIndTime += meshFromDepthShader.getElapsedTime();

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.2f}ms", totalRenderTime);
            spdlog::info("Create Mesh Time: {:.2f}ms", totalGenMeshTime);
            spdlog::info("  Create Vert/Ind Time: {:.2f}ms", totalCreateVertIndTime);
            spdlog::info("Frame Size: {:.3f}MB", ((float)(windowSize.x * windowSize.y * sizeof(uint32_t)) / 8) / BYTES_IN_MB);

            preventCopyingLocalPose = false;
            rerender = false;
        }

        nodeWireframe.visible = showWireframe;
        nodePointCloud.visible = showDepth;

        double startTime = window->getTime();

        // render generated meshes
        renderStats = renderer.drawObjects(scene, camera);

        toneMapper.enableToneMapping(true);
        toneMapper.drawToRenderTarget(remoteRenderer, renderTarget);
        if (animator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", (window->getTime() - startTime) * MILLISECONDS_IN_SECOND);
        }

        if ((animationFileIn && animator.running) || recording) {
            recorder.captureFrame(camera);
        }
        if (animationFileIn && !animator.running) {
            recorder.captureFrame(camera); // capture final frame
            recorder.stop();
            window->close();

            double avgPosError, avgRotError, avgTimeError, stdPosError, stdRotError, stdTimeError;
            poseSendRecvSimulator.getAvgErrors(avgPosError, avgRotError, avgTimeError, stdPosError, stdRotError, stdTimeError);
            spdlog::info("Pose Error: Pos ({:.2f}±{:.2f}), Rot ({:.2f}±{:.2f}), RTT ({:.2f}±{:.2f})",
                        avgPosError, stdPosError, avgRotError, stdRotError, avgTimeError, stdTimeError);
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
