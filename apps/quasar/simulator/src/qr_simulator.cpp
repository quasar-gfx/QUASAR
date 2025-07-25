#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>
#include <Renderers/DepthPeelingRenderer.h>

#include <HoleFiller.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <QRSimulator.h>

#include <PoseSendRecvSimulator.h>

#define REF_FRAME_PERIOD 5

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "QUASAR Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of local renderer", {'s', "size"}, "1920x1080");
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
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<int> maxLayersIn(parser, "layers", "Max layers", {'n', "max-layers"}, 4);
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

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    }

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

    int maxLayers = args::get(maxLayersIn);
    int maxViews = maxLayers + 1;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DepthPeelingRenderer remoteRendererDP(config, maxLayers, true);
    DeferredRenderer remoteRenderer(config);

    // "Remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCameraCenter(remoteRendererDP.width, remoteRendererDP.height);
    PerspectiveCamera remoteCameraWideFov(remoteRenderer.width, remoteRenderer.height);

    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCameraCenter);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCameraCenter.setFovyDegrees(remoteFOV);

     // Make last camera have a larger fov
    float remoteFOVWide = args::get(remoteFOVWideIn);
    remoteCameraWideFov.setFovyDegrees(remoteFOVWide);

    // "Local" scene with all the meshLayers
    Scene localScene;
    localScene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCameraCenter.getViewMatrix());

    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);
    FrameGenerator frameGenerator(remoteRenderer, remoteScene, quadsGenerator, meshFromQuads);
    QRSimulator quasarSimulator(remoteCameraCenter, maxViews, quadsGenerator, meshFromQuads, frameGenerator);

    quasarSimulator.addMeshesToScene(localScene);

    // Shaders
    ComputeShader meshFromDepthShader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_DEPTH_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_DEPTH_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    // Post processing
    HoleFiller holeFiller;

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, holeFiller, outputPath, config.targetFramerate);
    CameraAnimator cameraAnimator(cameraPathFile);

    if (saveImages) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();
    }

    if (cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        cameraAnimator.copyPoseToCamera(remoteCameraCenter);
    }

    bool generateRefFrame = true;
    bool generateResFrame = false;
    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewBox = !cameraPathFileIn;
    float viewSphereDiameter = args::get(viewSphereDiameterIn);
    remoteRendererDP.setViewSphereDiameter(viewSphereDiameter);

    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps
    double rerenderInterval = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = posePredictionIn;
    bool poseSmoothing = poseSmoothingIn;
    PoseSendRecvSimulator poseSendRecvSimulator({
        .networkLatencyMs = networkLatency,
        .networkJitterMs = networkJitter,
        .renderTimeMs = rerenderInterval / MILLISECONDS_IN_SECOND,
        .posePrediction = posePrediction,
        .poseSmoothing = poseSmoothing,
    });

    bool* showLayers = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showLayers[i] = true;
    }

    RenderStats renderStats;
    bool recording = false;
    std::vector<uint> numVertices(maxViews);
    std::vector<uint> numIndicies(maxViews);
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = !saveImages;
        static bool showLayerPreviews = false;
        static bool showFrameCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool showFramePreviewWindow = false;
        static char fileNameBase[256] = "screenshot";
        static bool saveAsHDR = false;
        static bool showRecordWindow = false;
        static int recordingFormatIndex = 0;
        static char recordingDirBase[256] = "recordings";

        static bool showSkyBox = true;

        for (int view = 0; view < maxViews; view++) {
            if (!showLayers[view]) {
                continue;
            }

            auto meshBufferSizes = frameGenerator.meshFromQuads.getBufferSizes();
            numVertices[view] = meshBufferSizes.numVertices;
            numIndicies[view] = meshBufferSizes.numIndices;
        }

        ImGui::NewFrame();

        uint flags = 0;
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
            ImGui::MenuItem("Frame Capture", 0, &showFrameCaptureWindow);
            ImGui::MenuItem("Record", 0, &showRecordWindow);
            ImGui::MenuItem("Mesh Capture", 0, &showMeshCaptureWindow);
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::MenuItem("Intermediate RT Previews", 0, &showFramePreviewWindow);
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

            uint totalTriangles = 0;
            for (int view = 0; view < maxViews; view++) {
                totalTriangles += numIndicies[view] / 3;
            }
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

                float proxySizeMB = static_cast<float>(quasarSimulator.stats.totalProxies * sizeof(QuadMapDataPacked)) / BYTES_PER_MEGABYTE;
                float depthOffsetSizeMB = static_cast<float>(quasarSimulator.stats.totalDepthOffsets * sizeof(uint16_t)) / BYTES_PER_MEGABYTE;
                ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", quasarSimulator.stats.totalProxies, proxySizeMB);
                ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", quasarSimulator.stats.totalDepthOffsets, depthOffsetSizeMB);

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
                    localScene.envCubeMap = showSkyBox ? remoteScene.envCubeMap : nullptr;
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

            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                preventCopyingLocalPose = true;
                generateRefFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                generateRefFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                if (ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator.params.correctOrientation)) {
                    preventCopyingLocalPose = true;
                    generateRefFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Depth Threshold", &quadsGenerator.params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                    preventCopyingLocalPose = true;
                    generateRefFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Angle Threshold", &quadsGenerator.params.angleThreshold, 0.1f, 0.0f, 180.0f)) {
                    preventCopyingLocalPose = true;
                    generateRefFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Flatten Threshold", &quadsGenerator.params.flattenThreshold, 0.001f, 0.0f, 1.0f)) {
                    preventCopyingLocalPose = true;
                    generateRefFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Similarity Threshold", &quadsGenerator.params.proxySimilarityThreshold, 0.001f, 0.0f, 2.0f)) {
                    preventCopyingLocalPose = true;
                    generateRefFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragInt("Force Merge Iterations", &quadsGenerator.params.maxIterForceMerge, 1, 0, quadsGenerator.numQuadMaps/2)) {
                    preventCopyingLocalPose = true;
                    generateRefFrame = true;
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
                rerenderInterval = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Send Reference Frame", ImVec2(buttonWidth, 0))) {
                generateRefFrame = true;
                runAnimations = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Send Residual Frame", ImVec2(buttonWidth, 0))) {
                generateResFrame = true;
                runAnimations = true;
            }

            ImGui::Separator();

            if (ImGui::DragFloat("View Sphere Diameter", &viewSphereDiameter, 0.025f, 0.1f, 1.5f)) {
                preventCopyingLocalPose = true;
                generateRefFrame = true;
                runAnimations = false;
                remoteRendererDP.setViewSphereDiameter(viewSphereDiameter);
            }

            ImGui::Checkbox("Restrict Movement to View Sphere", &restrictMovementToViewBox);

            ImGui::Separator();

            const int columns = 3;
            for (int view = 0; view < maxViews; view++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(view)).c_str(), &showLayers[view]);
                if ((view + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();
        }

        if (showLayerPreviews) {
            flags = ImGuiWindowFlags_AlwaysAutoResize;

            const int texturePreviewSize = (windowSize.x * 0.8) / maxViews;

            int rowSize = (maxViews + 1) / 2;
            for (int view = 0; view < maxViews; view++) {
                int viewIdx = maxViews - view - 1;
                if (showLayers[viewIdx]) {
                    int row = view / rowSize;
                    int col = view % rowSize;

                    ImGui::SetNextWindowPos(
                        ImVec2(windowSize.x - (col + 1) * texturePreviewSize - 30, 40 + row * (texturePreviewSize + 20)),
                        ImGuiCond_FirstUseEver
                    );

                    ImGui::Begin(("View " + std::to_string(viewIdx)).c_str(), 0, flags);
                    if (viewIdx == 0) {
                        ImGui::Image((void*)(intptr_t)(quasarSimulator.refFrameRT.colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    else {
                        ImGui::Image((void*)(intptr_t)(quasarSimulator.frameRTsHidLayer[viewIdx-1].colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    ImGui::End();
                }
            }
        }

        if (showFrameCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showFrameCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            Path basePath = outputPath / fileNameBase;

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                Path mainPath = basePath.appendToName("." + time);
                recorder.saveScreenshotToFile(mainPath, saveAsHDR);

                for (int view = 0; view < maxViews - 1; view++) {
                    Path viewPath = basePath.appendToName(".view" + std::to_string(view + 1) + "." + time);
                    if (saveAsHDR) {
                        quasarSimulator.frameRTsHidLayer[view].saveColorAsHDR(viewPath.withExtension(".hdr"));
                    }
                    else {
                        quasarSimulator.frameRTsHidLayer[view].saveColorAsPNG(viewPath.withExtension(".png"));
                    }
                }
            }

            ImGui::End();
        }

        if (showRecordWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Record", &showRecordWindow);

            if (recording) {
                ImGui::TextColored(ImVec4(1,0,0,1), "Recording in progress...");
            }

            ImGui::Text("Output Directory:");
            ImGui::InputText("##output directory", recordingDirBase, IM_ARRAYSIZE(recordingDirBase));

            ImGui::Text("FPS:");
            if (ImGui::InputInt("##fps", &recorder.targetFrameRate)) {
                recorder.setTargetFrameRate(recorder.targetFrameRate);
            }

            ImGui::Text("Format:");
            if (ImGui::Combo("##format", &recordingFormatIndex, recorder.getFormatCStrArray(), recorder.getFormatCount())) {
                Recorder::OutputFormat selectedFormat = Recorder::OutputFormat::MP4;
                switch (recordingFormatIndex) {
                    case 0: selectedFormat = Recorder::OutputFormat::MP4; break;
                    case 1: selectedFormat = Recorder::OutputFormat::PNG; break;
                    case 2: selectedFormat = Recorder::OutputFormat::JPG; break;
                    default: break;
                }
                recorder.setFormat(selectedFormat);
            }

            if (ImGui::Button("Start")) {
                recording = true;
                std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
                Path recordingDir = (outputPath / recordingDirBase).appendToName("." + time);
                recorder.setOutputPath(recordingDir);
                recorder.start();
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop")) {
                recorder.stop();
                recording = false;
            }

            ImGui::End();
        }

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                generateRefFrame = true;
                runAnimations = false;
                saveToFile = true;
            }

            ImGui::End();
        }

        if (showFramePreviewWindow) {
            flags = 0;
            ImGui::Begin("FrameRenderTarget Color", 0, flags);
            ImGui::Image((void*)(intptr_t)(quasarSimulator.refFrameRT.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("FrameRenderTarget Mask Color", 0, flags);
            ImGui::Image((void*)(intptr_t)(quasarSimulator.maskFrameRT.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        remoteRendererDP.setWindowSize(width, height);
        renderer.setWindowSize(width, height);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    double totalDT = 0.0;
    double lastRenderTime = -INFINITY;
    bool updateClient = !saveImages;
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

        if (rerenderInterval > 0.0 && (now - lastRenderTime) >= (rerenderInterval - 1.0) / MILLISECONDS_IN_SECOND) {
            generateRefFrame = (frameCounter++) % REF_FRAME_PERIOD == 0; // insert Reference Frame every REF_FRAME_PERIOD frames
            generateResFrame = !generateRefFrame;
        }
        if (generateRefFrame || generateResFrame) {
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
                    remoteCameraCenter.setViewMatrix(clientPosePred.mono.view);
                }
                // If we do not have a new pose, just send a new frame with the old pose
            }

            // Update wide fov camera
            remoteCameraWideFov.setViewMatrix(remoteCameraCenter.getViewMatrix());

            quasarSimulator.generateFrame(
                remoteCameraCenter, remoteCameraWideFov, remoteScene,
                remoteRenderer, remoteRendererDP, generateResFrame,
                showNormals, showDepth);

            std::string frameType = generateRefFrame ? "RefFrame" : "ResFrame";
            spdlog::info("======================================================");
            spdlog::info("Rendering Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalRenderTime);
            spdlog::info("Create Proxies Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalCreateProxiesTime);
            spdlog::info("  Gen Quad Map Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalGenQuadMapTime);
            spdlog::info("  Simplify Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalSimplifyTime);
            spdlog::info("  Gather Quads Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalGatherQuadsTime);
            spdlog::info("Create Mesh Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalCreateMeshTime);
            spdlog::info("  Append Quads Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalAppendQuadsTime);
            spdlog::info("  Fill Output Quads Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalFillQuadsIndiciesTime);
            spdlog::info("  Create Vert/Ind Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalCreateVertIndTime);
            spdlog::info("Compress Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalCompressTime);
            if (showDepth) spdlog::info("Gen Depth Time ({}): {:.3f}ms", frameType, quasarSimulator.stats.totalGenDepthTime);
            spdlog::info("Frame Size: {:.3f}MB", quasarSimulator.stats.compressedSizeBytes / BYTES_PER_MEGABYTE);
            spdlog::info("Num Proxies: {}Proxies", quasarSimulator.stats.totalProxies);

            // Save to file if requested
            if (saveToFile) {
                quasarSimulator.saveToFile(outputPath);
            }

            preventCopyingLocalPose = false;
            generateRefFrame = false;
            generateResFrame = false;
            saveToFile = false;
        }

        poseSendRecvSimulator.update(now);

        // Hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showLayer = showLayers[view];

            if (view == 0) {
                // Show previous mesh
                quasarSimulator.refFrameNodesLocal[quasarSimulator.currMeshIndex].visible = false;
                quasarSimulator.refFrameNodesLocal[quasarSimulator.prevMeshIndex].visible = showLayer;
                quasarSimulator.refFrameWireframesLocal[quasarSimulator.currMeshIndex].visible = false;
                quasarSimulator.refFrameWireframesLocal[quasarSimulator.prevMeshIndex].visible = showLayer && showWireframe;
                quasarSimulator.depthNode.visible = showLayer && showDepth;
            }
            else {
                quasarSimulator.nodesHidLayer[view-1].visible = showLayer;
                quasarSimulator.wireframesHidLayer[view-1].visible = showLayer && showWireframe;
                quasarSimulator.depthNodesHidLayer[view-1].visible = showLayer && showDepth;
            }
        }
        quasarSimulator.maskFrameWireframeNodesLocal.visible = quasarSimulator.maskFrameNode.visible && showWireframe;

        if (restrictMovementToViewBox) {
            glm::vec3 remotePosition = remoteCameraCenter.getPosition();
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

        double startTime = window->getTime();

        // Render all objects in scene
        renderStats = renderer.drawObjects(localScene, camera);

        // Render to screen
        holeFiller.enableToneMapping(!showNormals);
        holeFiller.setDepthThreshold(quadsGenerator.params.depthThreshold);
        holeFiller.drawToScreen(renderer);
        if (!updateClient) {
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCameraCenter);

        if (cameraPathFileIn) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.running) {
                poseSendRecvSimulator.printErrors();
                recorder.stop();
                window->close();
            }
        }
        else if (recording) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
