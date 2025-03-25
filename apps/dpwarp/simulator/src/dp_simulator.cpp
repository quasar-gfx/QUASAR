#include <filesystem>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>
#include <Renderers/DepthPeelingRenderer.h>

#include <BlurEdges.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <PoseSendRecvSimulator.h>

#define IFRAME_PERIOD 5

const std::vector<glm::vec4> colors = {
    glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
    glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
    glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "Depth Peeling Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of local renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
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

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCameraCenter(remoteRendererDP.width, remoteRendererDP.height);
    PerspectiveCamera remoteCameraCenterPrev(remoteRendererDP.width, remoteRendererDP.height);
    PerspectiveCamera remoteCameraWideFov(remoteRenderer.width, remoteRenderer.height);

    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCameraCenter);
    remoteCameraWideFov.setViewMatrix(remoteCameraCenter.getViewMatrix());
    remoteCameraCenterPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());

    float remoteFOV = args::get(remoteFOVIn);
    remoteCameraCenter.setFovyDegrees(remoteFOV);
    remoteCameraCenterPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());

     // make last camera have a larger fov
    float remoteFOVWide = args::get(remoteFOVWideIn);
    remoteCameraWideFov.setFovyDegrees(remoteFOVWide);

    const unsigned int numViewsWithoutCenter = maxViews - 1;

    // "local" scene with all the meshLayers
    Scene localScene;
    localScene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCameraCenter.getViewMatrix());

    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);
    FrameGenerator frameGenerator(remoteRenderer, remoteScene, quadsGenerator, meshFromQuads);

    // center RTs
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
    GBuffer gBufferCenterRT(rtParams);
    GBuffer gBufferCenterMaskRT(rtParams);
    GBuffer gBufferCenterTempRT(rtParams);

    GBuffer gBufferWideFovRT(rtParams);

    // hidden layers and wide fov RTs
    std::vector<GBuffer> gBufferHiddenRTs; gBufferHiddenRTs.reserve(numViewsWithoutCenter);
    for (int views = 0; views < numViewsWithoutCenter; views++) {
        if (views == numViewsWithoutCenter - 1) {
            rtParams.width /= 2; rtParams.height /= 2;
        }
        gBufferHiddenRTs.emplace_back(rtParams);
    }

    unsigned int maxVertices = MAX_NUM_PROXIES * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    // main view scenes and meshes
    std::vector<Scene> meshScenesCenter(2);
    int currMeshIndex = 0, prevMeshIndex = 1;

    std::vector<Mesh> meshesCenter; meshesCenter.reserve(2);
    std::vector<Node> nodeMeshesCenter; nodeMeshesCenter.reserve(2);
    std::vector<Node> nodeMeshesLocal; nodeMeshesLocal.reserve(2);
    std::vector<Node> nodeWireframesCenter; nodeWireframesCenter.reserve(2);

    MeshSizeCreateParams meshParams = {
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = new QuadMaterial({ .baseColorTexture = &gBufferCenterRT.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    };
    for (int i = 0; i < 2; i++) {
        meshesCenter.emplace_back(meshParams);

        nodeMeshesCenter.emplace_back(&meshesCenter[i]);
        nodeMeshesCenter[i].frustumCulled = false;
        meshScenesCenter[i].addChildNode(&nodeMeshesCenter[i]);

        nodeMeshesLocal.emplace_back(&meshesCenter[i]);
        nodeMeshesLocal[i].frustumCulled = false;
        localScene.addChildNode(&nodeMeshesLocal[i]);

        nodeWireframesCenter.emplace_back(&meshesCenter[i]);
        nodeWireframesCenter[i].frustumCulled = false;
        nodeWireframesCenter[i].wireframe = true;
        nodeWireframesCenter[i].visible = false;
        nodeWireframesCenter[i].overrideMaterial = new QuadMaterial({ .baseColor = colors[0] });
        localScene.addChildNode(&nodeWireframesCenter[i]);
    }

    meshParams.material = new QuadMaterial({ .baseColorTexture = &gBufferCenterMaskRT.colorBuffer });
    Mesh meshMask(meshParams);
    Node nodeMask(&meshMask);
    nodeMask.frustumCulled = false;

    Node nodeMaskWireframe(&meshMask);
    nodeMaskWireframe.frustumCulled = false;
    nodeMaskWireframe.wireframe = true;
    nodeMaskWireframe.visible = false;
    nodeMaskWireframe.overrideMaterial = new QuadMaterial({ .baseColor = colors[colors.size()-1] });

    localScene.addChildNode(&nodeMask);
    localScene.addChildNode(&nodeMaskWireframe);

    MeshSizeCreateParams meshDepthParams = {
        .maxVertices = maxVerticesDepth,
        .usage = GL_DYNAMIC_DRAW
    };
    meshDepthParams.material = new UnlitMaterial({ .baseColor = colors[0] });
    Mesh meshDepthCenter = Mesh(meshDepthParams);
    Node nodeDepth = Node(&meshDepthCenter);
    nodeDepth.frustumCulled = false;
    nodeDepth.visible = false;
    nodeDepth.primativeType = GL_POINTS;
    localScene.addChildNode(&nodeDepth);

    // hidden layers and wide fov scenes and meshes
    std::vector<Mesh> meshLayers; meshLayers.reserve(numViewsWithoutCenter);
    std::vector<Mesh> meshDepths; meshDepths.reserve(numViewsWithoutCenter);

    std::vector<Node> nodeLayers; nodeLayers.reserve(numViewsWithoutCenter);
    std::vector<Node> nodeWireframes; nodeWireframes.reserve(numViewsWithoutCenter);
    std::vector<Node> nodeDepths; nodeDepths.reserve(numViewsWithoutCenter);

    for (int view = 0; view < numViewsWithoutCenter; view++) {
        if (view == numViewsWithoutCenter - 1) {
            meshParams.material = new QuadMaterial({ .baseColorTexture = &gBufferWideFovRT.colorBuffer });
        }
        else {
            meshParams.material = new QuadMaterial({ .baseColorTexture = &gBufferHiddenRTs[view].colorBuffer });
        }
        meshLayers.emplace_back(meshParams);

        nodeLayers.emplace_back(&meshLayers[view]);
        nodeLayers[view].frustumCulled = false;
        localScene.addChildNode(&nodeLayers[view]);

        const glm::vec4 &color = colors[(view + 1) % colors.size()];

        nodeWireframes.emplace_back(&meshLayers[view]);
        nodeWireframes[view].frustumCulled = false;
        nodeWireframes[view].wireframe = true;
        nodeWireframes[view].overrideMaterial = new QuadMaterial({ .baseColor = color });
        localScene.addChildNode(&nodeWireframes[view]);

        meshDepthParams.material = new UnlitMaterial({ .baseColor = color });
        meshDepths.emplace_back(meshDepthParams);

        nodeDepths.emplace_back(&meshDepths[view]);
        nodeDepths[view].frustumCulled = false;
        nodeDepths[view].primativeType = GL_POINTS;
        localScene.addChildNode(&nodeDepths[view]);
    }

    // add all meshes to cover everything
    Scene meshScene;
    std::vector<Node> nodeMeshes; nodeMeshes.reserve(2);
    for (int i = 0; i < 2; i++) {
        nodeMeshes.emplace_back(&meshesCenter[i]);
        nodeMeshes[i].frustumCulled = false;
        meshScene.addChildNode(&nodeMeshes[i]);
    }
    for (int i = 0; i < numViewsWithoutCenter-1; i++) {
        meshScene.addChildNode(&nodeLayers[i]);
    }
    meshScene.addChildNode(&nodeMask);

    // shaders
    ComputeShader meshFromDepthShader({
        .computeCodeData = SHADER_COMMON_MESHFROMDEPTH_COMP,
        .computeCodeSize = SHADER_COMMON_MESHFROMDEPTH_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    // post processing
    BlurEdges blurEdges;
    ShowNormalsEffect showNormalsEffect;

    Recorder recorder(renderer, blurEdges, dataPath, config.targetFramerate);
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

    bool generateIFrame = true;
    bool generatePFrame = false;
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
    bool posePrediction = true;
    PoseSendRecvSimulator poseSendRecvSimulator(networkLatency, networkJitter, rerenderInterval / MILLISECONDS_IN_SECOND);

    bool* showLayers = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showLayers[i] = true;
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
        static bool showLayerPreviews = false;
        static bool showCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool showGBufferPreviewWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps

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
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::MenuItem("GBuffer Preview", 0, &showGBufferPreviewWindow);
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
                generateIFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                if (ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator.correctOrientation)) {
                    preventCopyingLocalPose = true;
                    generateIFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Depth Threshold", &quadsGenerator.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                    preventCopyingLocalPose = true;
                    generateIFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Angle Threshold", &quadsGenerator.angleThreshold, 0.1f, 0.0f, 180.0f)) {
                    preventCopyingLocalPose = true;
                    generateIFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Flat Threshold", &quadsGenerator.flatThreshold, 0.001f, 0.0f, 1.0f)) {
                    preventCopyingLocalPose = true;
                    generateIFrame = true;
                    runAnimations = false;
                }

                if (ImGui::DragFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.001f, 0.0f, 10.0f)) {
                    preventCopyingLocalPose = true;
                    generateIFrame = true;
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

            if (ImGui::Checkbox("Pose Prediction Enabled", &posePrediction)) {
                poseSendRecvSimulator.setPosePrediction(posePrediction);
            }

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderInterval = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Send I-Frame", ImVec2(buttonWidth, 0))) {
                generateIFrame = true;
                runAnimations = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Send P-Frame", ImVec2(buttonWidth, 0))) {
                generatePFrame = true;
                runAnimations = true;
            }

            ImGui::Separator();

            if (ImGui::DragFloat("View Sphere Diameter", &viewSphereDiameter, 0.05f, 0.1f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
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
                        ImGui::Image((void*)(intptr_t)(gBufferCenterRT.colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    else {
                        ImGui::Image((void*)(intptr_t)(gBufferHiddenRTs[viewIdx-1].colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    }
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
            std::string fileName = dataPath + std::string(fileNameBase) + "." + time;

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);

                for (int view = 0; view < numViewsWithoutCenter; view++) {
                    fileName = dataPath + std::string(fileNameBase) + ".view" + std::to_string(view+1) + "." + time;
                    if (saveAsHDR) {
                        gBufferHiddenRTs[view].saveColorAsHDR(fileName + ".hdr");
                    }
                    else {
                        gBufferHiddenRTs[view].saveColorAsPNG(fileName + ".png");
                    }
                }
            }

            ImGui::End();
        }

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            if (ImGui::Button("Save Mesh")) {
                for (int view = 0; view < maxViews; view++) {
                    std::string verticesFileName = dataPath + "vertices" + std::to_string(view) + ".bin";
                    std::string indicesFileName = dataPath + "indices" + std::to_string(view) + ".bin";

                    // save vertexBuffer
                    meshLayers[view].vertexBuffer.bind();
                    std::vector<QuadVertex> vertices = meshLayers[view].vertexBuffer.getData<QuadVertex>();
                    std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), numVertices[view] * sizeof(QuadVertex));
                    verticesFile.close();
                    spdlog::info("Saved {} vertices ({:.3f} MB) for layer {}",
                                 numVertices[view], static_cast<float>(numVertices[view] * sizeof(QuadVertex)) / BYTES_IN_MB, view);

                    // save indexBuffer
                    meshLayers[view].indexBuffer.bind();
                    std::vector<unsigned int> indices = meshLayers[view].indexBuffer.getData<unsigned int>();
                    std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), numIndicies[view] * sizeof(unsigned int));
                    indicesFile.close();
                    spdlog::info("Saved {} indicies ({:.3f} MB) for layer {}",
                                 numIndicies[view], static_cast<float>(numIndicies[view] * sizeof(unsigned int)) / BYTES_IN_MB, view);

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    gBufferHiddenRTs[view].saveColorAsPNG(colorFileName);
                }
            }

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
                saveToFile = true;
            }

            ImGui::End();
        }

        if (showGBufferPreviewWindow) {
            flags = 0;
            ImGui::Begin("GBuffer Color", 0, flags);
            ImGui::Image((void*)(intptr_t)(gBufferCenterRT.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("GBuffer Mask Color", 0, flags);
            ImGui::Image((void*)(intptr_t)(gBufferCenterMaskRT.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        remoteRendererDP.setWindowSize(width, height);
        renderer.setWindowSize(width, height);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    double lastRenderTime = 0.0;
    bool updateClient = !saveImages;
    int frameCounter = 0;
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
            generateIFrame = (++frameCounter) % IFRAME_PERIOD == 0; // insert I-Frame every IFRAME_PERIOD frames
            generatePFrame = !generateIFrame;
            // generateIFrame = true;
            runAnimations = true;
            lastRenderTime = now;
        }
        if (generateIFrame || generatePFrame) {
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
            double totalCompressTime = 0.0;

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

                    // update wide fov camera
                    remoteCameraWideFov.setViewMatrix(remoteCameraCenter.getViewMatrix());
                }
                // if we do not have a new pose, just send a new frame with the old pose
            }

            // render remote scene with multiple layers
            remoteRendererDP.drawObjects(remoteScene, remoteCameraCenter);
            totalRenderTime += timeutils::secondsToMillis(window->getTime() - startTime);

            for (int view = 0; view < maxViews; view++) {
                int hiddenIndex = view - 1;
                auto& remoteCameraToUse = (view == 0 && generatePFrame) ? remoteCameraCenterPrev :
                                            ((view != maxViews - 1) ? remoteCameraCenter : remoteCameraWideFov);

                auto& gBufferToUse = (view == 0) ? gBufferCenterRT : gBufferHiddenRTs[hiddenIndex];

                auto& meshToUse = (view == 0) ? meshesCenter[currMeshIndex] : meshLayers[hiddenIndex];
                auto& meshToUseDepth = (view == 0) ? meshDepthCenter : meshDepths[hiddenIndex];

                startTime = window->getTime();
                if (view == 0) {
                    remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
                    if (!showNormals) {
                        remoteRenderer.copyToGBuffer(gBufferToUse);
                    }
                    else {
                        showNormalsEffect.drawToRenderTarget(remoteRenderer, gBufferToUse);
                    }
                }
                else if (view != maxViews - 1) {
                    // copy to render target
                    if (!showNormals) {
                        remoteRendererDP.peelingLayers[hiddenIndex+1].blitToGBuffer(gBufferToUse);
                    }
                    else {
                        showNormalsEffect.drawToRenderTarget(remoteRendererDP, gBufferToUse);
                    }
                }
                // wide fov camera
                else {
                    // draw old center mesh at new remoteCamera view, filling stencil buffer with 1
                    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                    nodeMeshes[currMeshIndex].visible = false;
                    nodeMeshes[prevMeshIndex].visible = true;
                    remoteRenderer.drawObjectsNoLighting(meshScene, remoteCameraToUse);

                    // render remoteScene using stencil buffer as a mask
                    // at values where stencil buffer is not 1, remoteScene should render
                    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                    remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                    remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    remoteRenderer.pipeline.stencilState.restoreStencilState();

                    if (!showNormals) {
                        remoteRenderer.copyToGBuffer(gBufferToUse);
                        remoteRenderer.copyToGBuffer(gBufferWideFovRT);
                    }
                    else {
                        showNormalsEffect.drawToRenderTarget(remoteRenderer, gBufferToUse);
                        showNormalsEffect.drawToRenderTarget(remoteRenderer, gBufferWideFovRT);
                    }
                }
                totalRenderTime += timeutils::secondsToMillis(window->getTime() - startTime);

                /*
                ============================
                Generate I-frame
                ============================
                */
                unsigned int numProxies = 0, numDepthOffsets = 0;
                quadsGenerator.expandEdges = false;
                if (view != 0) {
                    quadsGenerator.depthThreshold *= 10.0f;
                    quadsGenerator.flatThreshold *= 10.0f;
                    quadsGenerator.proxySimilarityThreshold *= 10.0f;
                }
                unsigned int numBytesIFrame = frameGenerator.generateIFrame(
                    gBufferToUse, remoteCameraToUse,
                    meshToUse,
                    numProxies, numDepthOffsets
                );
                if (!generatePFrame) {
                    compressedSize += numBytesIFrame;
                }
                if (!(view == 0 && generatePFrame) && showLayers[view]) {
                    totalProxies += numProxies;
                    totalDepthOffsets += numDepthOffsets;

                    totalCompressTime += frameGenerator.stats.timeToCompress;
                }
                if (view != 0) {
                    quadsGenerator.depthThreshold /= 10.0f;
                    quadsGenerator.flatThreshold /= 10.0f;
                    quadsGenerator.proxySimilarityThreshold /= 10.0f;
                }

                totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
                totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

                totalAppendProxiesMsTime += frameGenerator.stats.timeToAppendProxiesMs;
                totalFillQuadsIndiciesMsTime += frameGenerator.stats.timeToFillQuadIndicesMs;
                totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

                /*
                ============================
                Generate P-frame
                ============================
                */
                if (view == 0) {
                    if (generatePFrame) {
                        quadsGenerator.expandEdges = true;
                        compressedSize += frameGenerator.generatePFrame(
                            meshScenesCenter[currMeshIndex], meshScenesCenter[prevMeshIndex],
                            gBufferCenterTempRT, gBufferCenterMaskRT,
                            remoteCameraCenter, remoteCameraCenterPrev,
                            meshesCenter[currMeshIndex], meshMask,
                            numProxies, numDepthOffsets
                        );
                        totalProxies += numProxies;
                        totalDepthOffsets += numDepthOffsets;

                        totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                        totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                        totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
                        totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

                        totalAppendProxiesMsTime += frameGenerator.stats.timeToAppendProxiesMs;
                        totalFillQuadsIndiciesMsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
                        totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                        totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

                        totalCompressTime += frameGenerator.stats.timeToCompress;
                    }
                    nodeMask.visible = generatePFrame;
                    currMeshIndex = (currMeshIndex + 1) % 2;
                    prevMeshIndex = (prevMeshIndex + 1) % 2;

                    // only update the previous camera pose if we are not generating a P-Frame
                    if (generateIFrame) {
                        remoteCameraCenterPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());
                    }
                }

                if (saveToFile) {
                    unsigned int savedBytes;

                    startTime = window->getTime();
                    std::string quadsFileName = dataPath + "quads" + std::to_string(view) + ".bin";
                    savedBytes = quadsGenerator.saveToFile(quadsFileName);
                    spdlog::info("Saved {} quads ({:.3f} MB) in {:.3f}ms",
                                 numProxies, static_cast<float>(savedBytes) / BYTES_IN_MB,
                                    timeutils::secondsToMillis(window->getTime() - startTime));

                    startTime = window->getTime();
                    std::string depthOffsetsFileName = dataPath + "depthOffsets" + std::to_string(view) + ".bin";
                    savedBytes = quadsGenerator.saveDepthOffsetsToFile(depthOffsetsFileName);
                    spdlog::info("Saved {} depth offsets ({:.3f} MB) in {:.3f}ms",
                                 numDepthOffsets, static_cast<float>(savedBytes) / BYTES_IN_MB,
                                    timeutils::secondsToMillis(window->getTime() - startTime));

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    gBufferToUse.saveColorAsPNG(colorFileName);
                }

                // For debugging: Generate point cloud from depth map
                if (showDepth) {
                    const glm::vec2 gBufferSize = glm::vec2(gBufferToUse.width, gBufferToUse.height);

                    startTime = window->getTime();

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

                    totalGenDepthTime += timeutils::secondsToMillis(window->getTime() - startTime);
                }
            }

            std::string frameType = generateIFrame ? "I-Frame" : "P-Frame";
            spdlog::info("======================================================");
            spdlog::info("Rendering Time ({}): {:.3f}ms", frameType, totalRenderTime);
            spdlog::info("Create Proxies Time ({}): {:.3f}ms", frameType, totalCreateProxiesTime);
            spdlog::info("  Gen Quad Map Time ({}): {:.3f}ms", frameType, totalGenQuadMapTime);
            spdlog::info("  Simplify Time ({}): {:.3f}ms", frameType, totalSimplifyTime);
            spdlog::info("  Fill Quads Time ({}): {:.3f}ms", frameType, totalFillQuadsTime);
            spdlog::info("Create Mesh Time ({}): {:.3f}ms", frameType, totalCreateMeshTime);
            spdlog::info("  Append Quads Time ({}): {:.3f}ms", frameType, totalAppendProxiesMsTime);
            spdlog::info("  Fill Output Quads Time ({}): {:.3f}ms", frameType, totalFillQuadsIndiciesMsTime);
            spdlog::info("  Create Vert/Ind Time ({}): {:.3f}ms", frameType, totalCreateVertIndTime);
            spdlog::info("Compress Time ({}): {:.3f}ms", frameType, totalCompressTime);
            if (showDepth) spdlog::info("Gen Depth Time ({}): {:.3f}ms", frameType, totalGenDepthTime);
            spdlog::info("Frame Size: {:.3f}MB", static_cast<float>(compressedSize) / BYTES_IN_MB);
            spdlog::info("Num Proxies: {}Proxies", totalProxies);

            preventCopyingLocalPose = false;
            generateIFrame = false;
            generatePFrame = false;
            saveToFile = false;
        }

        poseSendRecvSimulator.update(now);

        // hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showLayer = showLayers[view];

            if (view == 0) {
                // show previous mesh
                nodeMeshesLocal[currMeshIndex].visible = false;
                nodeMeshesLocal[prevMeshIndex].visible = showLayer;
                nodeWireframesCenter[currMeshIndex].visible = false;
                nodeWireframesCenter[prevMeshIndex].visible = showLayer && showWireframe;
                nodeDepth.visible = showLayer && showDepth;
            }
            else {
                nodeLayers[view-1].visible = showLayer;
                nodeWireframes[view-1].visible = showLayer && showWireframe;
                nodeDepths[view-1].visible = showLayer && showDepth;
            }
        }
        nodeMaskWireframe.visible = nodeMask.visible && showWireframe;

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

        // render all objects in scene
        renderStats = renderer.drawObjects(localScene, camera);

        // render to screen
        blurEdges.enableToneMapping(!showNormals);
        blurEdges.setDepthThreshold(quadsGenerator.depthThreshold);
        blurEdges.drawToScreen(renderer);
        if (!updateClient) {
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCameraCenter);

        if ((cameraPathFileIn && cameraAnimator.running) || recording) {
            recorder.captureFrame(camera);
        }
        if (cameraPathFileIn && !cameraAnimator.running) {
            poseSendRecvSimulator.printErrors();
            recorder.captureFrame(camera);
            recorder.stop();
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
