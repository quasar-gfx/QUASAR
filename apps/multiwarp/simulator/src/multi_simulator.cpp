#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <PoseSendRecvSimulator.h>

#include <shaders_common.h>

using namespace quasar;

const std::vector<glm::vec4> colors = {
    glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
    glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
    glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
};

const std::vector<glm::vec3> offsets = {
    glm::vec3(-1.0f, +1.0f, -1.0f), // Top-left
    glm::vec3(+1.0f, +1.0f, -1.0f), // Top-right
    glm::vec3(+1.0f, -1.0f, -1.0f), // Bottom-right
    glm::vec3(-1.0f, -1.0f, -1.0f), // Bottom-left
    glm::vec3(-1.0f, +1.0f, +1.0f), // Top-left
    glm::vec3(+1.0f, +1.0f, +1.0f), // Top-right
    glm::vec3(+1.0f, -1.0f, +1.0f), // Bottom-right
    glm::vec3(-1.0f, -1.0f, +1.0f), // Bottom-left
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "Multi-Camera Simulator";

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
    std::string outputPath = args::get(outputPathIn);
    if (outputPath.back() != '/') {
        outputPath += "/";
    }
    // create data path if it doesn't exist
    if (!std::filesystem::exists(outputPath)) {
        std::filesystem::create_directories(outputPath);
    }

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

    // "remote" scene
    Scene remoteScene;
    std::vector<PerspectiveCamera> remoteCameras; remoteCameras.reserve(maxViews);
    for (int view = 0; view < maxViews; view++) {
        if (view == maxViews - 1) {
            remoteCameras.emplace_back(1280, 720);
        }
        remoteCameras.emplace_back(remoteRenderer.width, remoteRenderer.height);
    }
    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCameraCenter);

    float remoteFOV = args::get(remoteFOVIn);
    // make last camera have a larger fov
    float remoteFOVWide = args::get(remoteFOVWideIn);
    for (int view = 0; view < maxViews; view++) {
        if (view != maxViews - 1) {
            remoteCameras[view].setFovyDegrees(remoteFOV);
        }
        else {
            remoteCameras[view].setFovyDegrees(remoteFOVWide);
        }
    }

    // scene with all the meshViews
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCameraCenter.getViewMatrix());

    QuadsGenerator quadsGenerator(remoteWindowSize);
    // match QuadStream's parameters:
    quadsGenerator.expandEdges = true;
    quadsGenerator.depthThreshold = 1e-4f;
    quadsGenerator.flatThreshold = 1e-2f;
    quadsGenerator.proxySimilarityThreshold = 0.1f;
    MeshFromQuads meshFromQuads(remoteWindowSize);
    FrameGenerator frameGenerator(remoteRenderer, remoteScene, quadsGenerator, meshFromQuads);

    std::vector<FrameRenderTarget> gBufferRTs; gBufferRTs.reserve(maxViews);
    RenderTargetCreateParams rtParams = {
        .width = remoteWindowSize.x,
        .height = remoteWindowSize.y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    };
    for (int views = 0; views < maxViews; views++) {
        if (views == maxViews - 1) {
            rtParams.width = 1280; rtParams.height = 720;
        }
        gBufferRTs.emplace_back(rtParams);
    }

    unsigned int maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    Scene meshScene;
    std::vector<Mesh> meshViews; meshViews.reserve(maxViews);
    std::vector<Mesh> meshDepths; meshDepths.reserve(maxViews);

    std::vector<Node> nodes; nodes.reserve(maxViews);
    std::vector<Node> nodeMeshes; nodeMeshes.reserve(maxViews);
    std::vector<Node> nodeWireframes; nodeWireframes.reserve(maxViews);
    std::vector<Node> nodeDepths; nodeDepths.reserve(maxViews);

    MeshSizeCreateParams meshParams = {
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    };
    for (int view = 0; view < maxViews; view++) {
        meshParams.material = new QuadMaterial({ .baseColorTexture = &gBufferRTs[view].colorBuffer });
        meshParams.maxVertices = maxVertices / (view == 0 ? 1 : 2);
        meshParams.maxIndices = maxIndices / (view == 0 ? 1 : 2);
        meshViews.emplace_back(meshParams);

        nodes.emplace_back(&meshViews[view]);
        nodes[view].frustumCulled = false;
        scene.addChildNode(&nodes[view]);

        const glm::vec4 &color = colors[view % colors.size()];

        nodeWireframes.emplace_back(&meshViews[view]);
        nodeWireframes[view].frustumCulled = false;
        nodeWireframes[view].wireframe = true;
        nodeWireframes[view].overrideMaterial = new QuadMaterial({ .baseColor = color });
        scene.addChildNode(&nodeWireframes[view]);

        MeshSizeCreateParams meshDepthParams = {
            .maxVertices = maxVerticesDepth,
            .material = new QuadMaterial({ .baseColor = color }),
            .usage = GL_DYNAMIC_DRAW
        };
        meshDepths.emplace_back(meshDepthParams);

        nodeDepths.emplace_back(&meshDepths[view]);
        nodeDepths[view].frustumCulled = false;
        nodeDepths[view].primativeType = GL_POINTS;
        scene.addChildNode(&nodeDepths[view]);

        nodeMeshes.emplace_back(&meshViews[view]);
        nodeMeshes[view].frustumCulled = false;
        nodeMeshes[view].visible = (view == 0);
        meshScene.addChildNode(&nodeMeshes[view]);
    }

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
    ShowNormalsEffect showNormalsEffect;

    Recorder recorder(renderer, toneMapper, outputPath, config.targetFramerate);
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

    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewBox = !cameraPathFileIn;
    float viewBoxSize = args::get(viewBoxSizeIn);

    bool generateRemoteFrame = true;

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
        .poseSmoothing = poseSmoothing
    });

    bool* showViews = new bool[maxViews];
    for (int view = 0; view < maxViews; ++view) {
        showViews[view] = true;
    }

    unsigned int totalProxies = 0;
    unsigned int totalDepthOffsets = 0;

    RenderStats renderStats;
    bool recording = false;
    std::vector<unsigned int> numVertices(maxViews);
    std::vector<unsigned int> numIndicies(maxViews);
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = !saveImages;
        static bool showViewPreviews = false;
        static bool showCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps

        static bool showSkyBox = true;

        for (int view = 0; view < maxViews; view++) {
            if (!showViews[view]) {
                continue;
            }

            auto meshBufferSizes = frameGenerator.meshFromQuads.getBufferSizes();
            numVertices[view] = meshBufferSizes.numVertices;
            numIndicies[view] = meshBufferSizes.numIndices;
        }

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
            ImGui::MenuItem("Mesh Capture", 0, &showMeshCaptureWindow);
            ImGui::MenuItem("View Previews", 0, &showViewPreviews);
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

            unsigned int totalTriangles = 0;
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

            float proxySizeMb = static_cast<float>(totalProxies * sizeof(QuadMapDataPacked)) / BYTES_IN_MB;
            float depthOffsetSizeMb = static_cast<float>(totalDepthOffsets * sizeof(uint16_t)) / BYTES_IN_MB;
            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", totalProxies, proxySizeMb);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", totalDepthOffsets, depthOffsetSizeMb);

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

            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                preventCopyingLocalPose = true;
                generateRemoteFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                generateRemoteFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                if (ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator.correctOrientation)) {
                    preventCopyingLocalPose = true;
                    generateRemoteFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Depth Threshold", &quadsGenerator.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                    preventCopyingLocalPose = true;
                    generateRemoteFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Angle Threshold", &quadsGenerator.angleThreshold, 0.1f, 0.0f, 180.0f)) {
                    preventCopyingLocalPose = true;
                    generateRemoteFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Flat Threshold", &quadsGenerator.flatThreshold, 0.001f, 0.0f, 1.0f)) {
                    preventCopyingLocalPose = true;
                    generateRemoteFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.001f, 0.0f, 10.0f)) {
                    preventCopyingLocalPose = true;
                    generateRemoteFrame = true;
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

            if (ImGui::Button("Send Server Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                generateRemoteFrame = true;
                runAnimations = true;
            }

            ImGui::Separator();

            if (ImGui::DragFloat("View Box Size", &viewBoxSize, 0.05f, 0.1f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateRemoteFrame = true;
                runAnimations = false;
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

        if (showViewPreviews) {
            flags = ImGuiWindowFlags_AlwaysAutoResize;

            const int texturePreviewSize = (windowSize.x * 0.8) / maxViews;

            int rowSize = (maxViews + 1) / 2;
            for (int view = 0; view < maxViews; view++) {
                int viewIdx = maxViews - view - 1;
                if (showViews[viewIdx]) {
                    int row = view / rowSize;
                    int col = view % rowSize;

                    ImGui::SetNextWindowPos(
                        ImVec2(windowSize.x - (col + 1) * texturePreviewSize - 30, 40 + row * (texturePreviewSize + 20)),
                        ImGuiCond_FirstUseEver
                    );

                    ImGui::Begin(("View " + std::to_string(viewIdx)).c_str(), 0, flags);
                    ImGui::Image((void*)(intptr_t)(gBufferRTs[viewIdx].colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }
            }
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            std::string fileName = outputPath + std::string(fileNameBase) + "." + time;

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);

                for (int view = 1; view < maxViews; view++) {
                    fileName = outputPath + std::string(fileNameBase) + ".view" + std::to_string(view) + "." + time;
                    if (saveAsHDR) {
                        gBufferRTs[view].saveColorAsHDR(fileName + ".hdr");
                    }
                    else {
                        gBufferRTs[view].saveColorAsPNG(fileName + ".png");
                    }
                }
            }

            ImGui::End();
        }

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                generateRemoteFrame = true;
                runAnimations = false;
                saveToFile = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Save Mesh")) {
                for (int view = 0; view < maxViews; view++) {
                    std::string verticesFileName = outputPath + "vertices" + std::to_string(view) + ".bin";
                    std::string indicesFileName = outputPath + "indices" + std::to_string(view) + ".bin";

                    // save vertexBuffer
                    meshViews[view].vertexBuffer.bind();
                    std::vector<QuadVertex> vertices = meshViews[view].vertexBuffer.getData<QuadVertex>();
                    std::ofstream verticesFile(outputPath + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), numVertices[view] * sizeof(QuadVertex));
                    verticesFile.close();
                    spdlog::info("Saved {} vertices ({:.3f} MB) for view {}",
                                 numVertices[view], static_cast<float>(numVertices[view] * sizeof(QuadVertex)) / BYTES_IN_MB, view);

                    // save indexBuffer
                    meshViews[view].indexBuffer.bind();
                    std::vector<unsigned int> indices = meshViews[view].indexBuffer.getData<unsigned int>();
                    std::ofstream indicesFile(outputPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), numIndicies[view] * sizeof(unsigned int));
                    indicesFile.close();
                    spdlog::info("Saved {} indicies ({:.3f} MB) for view {}",
                                 numIndicies[view], static_cast<float>(numIndicies[view] * sizeof(unsigned int)) / BYTES_IN_MB, view);

                    // save color buffer
                    std::string colorFileName = outputPath + "color" + std::to_string(view) + ".png";
                    gBufferRTs[view].saveColorAsPNG(colorFileName);
                }
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

        if (rerenderInterval > 0.0 && (now - lastRenderTime) >= (rerenderInterval - 1.0) / MILLISECONDS_IN_SECOND) {
            generateRemoteFrame = true;
            lastRenderTime = now;
        }
        if (generateRemoteFrame) {
            double startTime = window->getTime();
            double totalRenderTime = 0.0;
            double totalCreateProxiesTime = 0.0;
            double totalGenQuadMapTime = 0.0;
            double totalSimplifyTime = 0.0;
            double totalFillQuadsTime = 0.0;
            double totalCreateMeshTime = 0.0;
            double totalAppendProxiesMsTime = 0.0;
            double totalFillQuadsIndiciesMsTime = 0.0;
            double totalCreateVertIndTime = 0.0;
            double totalGenDepthTime = 0.0;

            unsigned int compressedSize = 0;

            totalProxies = 0;
            totalDepthOffsets = 0;

            // "send" pose to the server. this will wait until latency+/-jitter ms have passed
            poseSendRecvSimulator.sendPose(camera, now);
            if (!preventCopyingLocalPose) {
                // "receive" a predicted pose to render a new frame. this will wait until latency+/-jitter ms have passed
                Pose clientPosePred;
                if (poseSendRecvSimulator.recvPoseToRender(clientPosePred, now)) {
                    remoteCameraCenter.setViewMatrix(clientPosePred.mono.view);
                }
                // if we do not have a new pose, just send a new frame with the old pose
            }

            // update other cameras in view box corners
            for (int view = 1; view < maxViews - 1; view++) {
                const glm::vec3 &offset = offsets[view - 1];
                const glm::vec3 &right = remoteCameraCenter.getRightVector();
                const glm::vec3 &up = remoteCameraCenter.getUpVector();
                const glm::vec3 &forward = remoteCameraCenter.getForwardVector();

                glm::vec3 worldOffset =
                    right   * offset.x * viewBoxSize / 2.0f +
                    up      * offset.y * viewBoxSize / 2.0f +
                    forward * offset.z * viewBoxSize / 2.0f;

                remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
                remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + worldOffset);
                remoteCameras[view].updateViewMatrix();
            }
            // update wide fov camera
            remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());

            for (int view = 0; view < maxViews; view++) {
                auto& remoteCameraToUse = remoteCameras[view];

                auto& gBufferToUse = gBufferRTs[view];

                auto& meshToUse = meshViews[view];
                auto& meshToUseDepth = meshDepths[view];

                startTime = window->getTime();

                // center view
                if (view == 0) {
                    // render all objects in remoteScene normally
                    remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
                }
                // other views
                else {
                    // make all previous meshViews visible and everything else invisible
                    for (int prevView = 1; prevView < maxViews; prevView++) {
                        meshScene.rootNode.children[prevView]->visible = (prevView < view);
                    }
                    // draw old meshViews at new remoteCamera view, filling stencil buffer with 1
                    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                    remoteRenderer.drawObjectsNoLighting(meshScene, remoteCameraToUse);

                    // render remoteScene using stencil buffer as a mask
                    // at values where stencil buffer is not 1, remoteScene should render
                    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                    remoteRenderer.pipeline.rasterState.polygonOffsetEnabled = false;
                    remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                    remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    remoteRenderer.pipeline.stencilState.restoreStencilState();
                }
                if (!showNormals) {
                    remoteRenderer.copyToFrameRT(gBufferToUse);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, gBufferToUse);
                }
                totalRenderTime += timeutils::secondsToMillis(window->getTime() - startTime);

                unsigned int numProxies = 0, numDepthOffsets = 0;
                compressedSize += frameGenerator.generateIFrame(
                    gBufferToUse, remoteCameraToUse,
                    meshToUse,
                    numProxies, numDepthOffsets,
                    false
                );
                if (showViews[view]) {
                    totalProxies += numProxies;
                    totalDepthOffsets += numDepthOffsets;
                }

                totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
                totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

                totalAppendProxiesMsTime += frameGenerator.stats.timeToAppendProxiesMs;
                totalFillQuadsIndiciesMsTime += frameGenerator.stats.timeToFillQuadIndicesMs;
                totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

                if (saveToFile) {
                    unsigned int savedBytes;

                    startTime = window->getTime();
                    std::string quadsFileName = outputPath + "quads" + std::to_string(view) + ".bin";
                    savedBytes = quadsGenerator.saveToFile(quadsFileName);
                    spdlog::info("Saved {} quads ({:.3f} MB) in {:.3f}ms",
                                 numProxies, static_cast<float>(savedBytes) / BYTES_IN_MB,
                                    timeutils::secondsToMillis(window->getTime() - startTime));

                    startTime = window->getTime();
                    std::string depthOffsetsFileName = outputPath + "depthOffsets" + std::to_string(view) + ".bin";
                    savedBytes = quadsGenerator.saveDepthOffsetsToFile(depthOffsetsFileName);
                    spdlog::info("Saved {} depth offsets ({:.3f} MB) in {:.3f}ms",
                                 numDepthOffsets, static_cast<float>(savedBytes) / BYTES_IN_MB,
                                    timeutils::secondsToMillis(window->getTime() - startTime));

                    // save color buffer
                    std::string colorFileName = outputPath + "color" + std::to_string(view) + ".png";
                    gBufferToUse.saveColorAsPNG(colorFileName);
                }

                // for debugging: Generate point cloud from depth map
                if (showDepth) {
                    const glm::vec2 gBufferSize = glm::vec2(gBufferToUse.width, gBufferToUse.height);

                    meshFromDepthShader.startTiming();

                    meshFromDepthShader.bind();
                    {
                        meshFromDepthShader.setTexture(gBufferToUse.depthStencilBuffer, 0);
                    }
                    {
                        meshFromDepthShader.setVec2("depthMapSize", gBufferSize);
                    }
                    {
                        meshFromDepthShader.setMat4("view", remoteCameraToUse.getViewMatrix());
                        meshFromDepthShader.setMat4("projection", remoteCameraToUse.getProjectionMatrix());
                        meshFromDepthShader.setMat4("viewInverse", remoteCameraToUse.getViewMatrixInverse());
                        meshFromDepthShader.setMat4("projectionInverse", remoteCameraToUse.getProjectionMatrixInverse());

                        meshFromDepthShader.setFloat("near", remoteCameraToUse.getNear());
                        meshFromDepthShader.setFloat("far", remoteCameraToUse.getFar());
                    }
                    {
                        meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshToUseDepth.vertexBuffer);
                        meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
                    }
                    meshFromDepthShader.dispatch((gBufferSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                 (gBufferSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                    meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                    meshFromDepthShader.endTiming();
                    totalGenDepthTime += meshFromDepthShader.getElapsedTime();
                }
            }

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.3f}ms", totalRenderTime);
            spdlog::info("Create Proxies Time: {:.3f}ms", totalCreateProxiesTime);
            spdlog::info("  Gen Quad Map Time: {:.3f}ms", totalGenQuadMapTime);
            spdlog::info("  Simplify Time: {:.3f}ms", totalSimplifyTime);
            spdlog::info("  Fill Quads Time: {:.3f}ms", totalFillQuadsTime);
            spdlog::info("Create Mesh Time: {:.3f}ms", totalCreateMeshTime);
            spdlog::info("  Append Quads Time: {:.3f}ms", totalAppendProxiesMsTime);
            spdlog::info("  Fill Output Quads Time: {:.3f}ms", totalFillQuadsIndiciesMsTime);
            spdlog::info("  Create Vert/Ind Time: {:.3f}ms", totalCreateVertIndTime);
            if (showDepth) spdlog::info("Gen Depth Time: {:.3f}ms", totalGenDepthTime);
            // QS has data structures that are 103 bits
            spdlog::info("Frame Size: {:.3f}MB", static_cast<float>(compressedSize) / BYTES_IN_MB * (103.0) / (8*sizeof(QuadMapDataPacked)));
            spdlog::info("Num Proxies: {}Proxies", totalProxies);

            preventCopyingLocalPose = false;
            generateRemoteFrame = false;
            saveToFile = false;
        }

        poseSendRecvSimulator.update(now);

        // hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showView = showViews[view];

            nodes[view].visible = showView;
            nodeWireframes[view].visible = showView && showWireframe;
            nodeDepths[view].visible = showView && showDepth;
        }

        if (restrictMovementToViewBox) {
            glm::vec3 remotePosition = remoteCameraCenter.getPosition();
            glm::vec3 position = camera.getPosition();
            // restrict camera position to be inside position±viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - viewBoxSize/2, remotePosition.x + viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewBoxSize/2, remotePosition.y + viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewBoxSize/2, remotePosition.z + viewBoxSize/2);
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        double startTime = window->getTime();

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        toneMapper.enableToneMapping(!showNormals);
        toneMapper.drawToScreen(renderer);
        if (!updateClient) {
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCameraCenter);

        if (cameraPathFileIn) {
            poseSendRecvSimulator.printErrors();
            recorder.captureFrame(camera);

            if (!cameraAnimator.running) {
                recorder.stop();
                window->close();
            }
        }
        else if (recording) {
            recorder.captureFrame(camera);
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
