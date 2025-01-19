#include <iostream>
#include <filesystem>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Recorder.h>
#include <Animator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <PoseSendRecvSimulator.h>

#include <shaders_common.h>

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
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'V', "vsync"}, true);
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::Flag saveImage(parser, "save", "Take screenshot and exit", {'I', "save-image"});
    args::ValueFlag<std::string> animationFileIn(parser, "path", "Path to camera animation file", {'A', "animation-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<int> maxLayersIn(parser, "layers", "Max layers", {'n', "max-layers"}, 4);
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
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    // assume remote window size is the same as local window size
    glm::uvec2 remoteWindowSize = glm::uvec2(config.width, config.height);

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

    int maxLayers = args::get(maxLayersIn);
    int maxViews = maxLayers + 1;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    ForwardRenderer remoteRenderer(config);
    DepthPeelingRenderer dpRenderer(config, maxLayers, true);
    ForwardRenderer wideFOVRenderer(config);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCameraCenter(dpRenderer.width, dpRenderer.height);
    PerspectiveCamera remoteCameraCenterPrev(dpRenderer.width, dpRenderer.height);
    PerspectiveCamera remoteCameraWideFov(wideFOVRenderer.width, wideFOVRenderer.height);
    remoteCameraWideFov.setFovyDegrees(120.0f); // make last camera have a larger fov
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCameraCenter);
    remoteCameraWideFov.setViewMatrix(remoteCameraCenter.getViewMatrix());
    remoteCameraCenterPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());

    const unsigned int numViewsWithoutCenter = maxViews - 1;

    // "local" scene with all the meshLayers
    Scene localScene;
    localScene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCameraCenter.getViewMatrix());

    FrameGenerator frameGenerator;
    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);
    MeshFromQuads meshFromQuadsMask(remoteWindowSize, MAX_NUM_PROXIES / 4);

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

    // hidden layers and wide fov RTs
    std::vector<GBuffer> gBufferHiddenRTs; gBufferHiddenRTs.reserve(numViewsWithoutCenter);
    for (int views = 0; views < numViewsWithoutCenter; views++) {
        if (views == numViewsWithoutCenter - 1) {
            rtParams.width /= 2; rtParams.height /= 2; // set to lower resolution for wide fov
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

    for (int i = 0; i < 2; i++) {
        MeshSizeCreateParams meshParams = {
            .numVertices = maxVertices,
            .numIndices = maxIndices,
            .material = new QuadMaterial({ .baseColorTexture = &gBufferCenterRT.colorBuffer }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        };
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
        nodeWireframesCenter[i].overrideMaterial = new UnlitMaterial({ .baseColor = colors[0] });
        localScene.addChildNode(&nodeWireframesCenter[i]);
    }

    Mesh meshMask({
        .numVertices = maxVertices,
        .numIndices = maxIndices,
        .material = new QuadMaterial({ .baseColorTexture = &gBufferCenterMaskRT.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    });
    Node nodeMask(&meshMask);
    nodeMask.frustumCulled = false;

    Node nodeMaskWireframe(&meshMask);
    nodeMaskWireframe.frustumCulled = false;
    nodeMaskWireframe.wireframe = true;
    nodeMaskWireframe.visible = false;
    nodeMaskWireframe.overrideMaterial = new UnlitMaterial({ .baseColor = colors[1] });

    localScene.addChildNode(&nodeMask);
    localScene.addChildNode(&nodeMaskWireframe);

    // hidden layers and wide fov scenes and meshes
    Scene meshScene;

    std::vector<Mesh> meshLayers; meshLayers.reserve(numViewsWithoutCenter);
    std::vector<Mesh> meshDepths; meshDepths.reserve(numViewsWithoutCenter);

    std::vector<Node> nodeLayers; nodeLayers.reserve(numViewsWithoutCenter);
    std::vector<Node> nodeWireframes; nodeWireframes.reserve(numViewsWithoutCenter);
    std::vector<Node> nodeDepths; nodeDepths.reserve(numViewsWithoutCenter);

    for (int view = 0; view < numViewsWithoutCenter; view++) {
        MeshSizeCreateParams meshParams = {
            .numVertices = maxVertices,
            .numIndices = maxIndices,
            .material = new QuadMaterial({ .baseColorTexture = &gBufferHiddenRTs[view].colorBuffer }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        };
        meshLayers.emplace_back(meshParams);

        nodeLayers.emplace_back(&meshLayers[view]);
        nodeLayers[view].frustumCulled = false;
        localScene.addChildNode(&nodeLayers[view]);

        const glm::vec4 &color = colors[(view + 2) % colors.size()];

        nodeWireframes.emplace_back(&meshLayers[view]);
        nodeWireframes[view].frustumCulled = false;
        nodeWireframes[view].wireframe = true;
        nodeWireframes[view].overrideMaterial = new UnlitMaterial({ .baseColor = color });
        localScene.addChildNode(&nodeWireframes[view]);

        MeshSizeCreateParams meshDepthParams = {
            .numVertices = maxVerticesDepth,
            .material = new UnlitMaterial({ .baseColor = color }),
            .usage = GL_DYNAMIC_DRAW
        };
        meshDepths.emplace_back(meshDepthParams);

        nodeDepths.emplace_back(&meshDepths[view]);
        nodeDepths[view].frustumCulled = false;
        nodeDepths[view].primativeType = GL_POINTS;
        localScene.addChildNode(&nodeDepths[view]);
    }

    // main mesh covers everything
    std::vector<Node> nodeMeshes; nodeMeshes.reserve(2);
    for (int i = 0; i < 2; i++) {
        nodeMeshes.emplace_back(&meshesCenter[i]);
        nodeMeshes[i].frustumCulled = false;
        meshScene.addChildNode(&nodeMeshes[i]);
    }
    meshScene.addChildNode(&nodeMask);

    // shaders
    Shader blurEdgesShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_COMMON_BLUREDGES_FRAG,
        .fragmentCodeSize = SHADER_COMMON_BLUREDGES_FRAG_len
    });

    Shader screenShaderNormals({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYNORMALS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYNORMALS_FRAG_len
    });

    ComputeShader meshFromDepthShader({
        .computeCodeData = SHADER_COMMON_MESHFROMDEPTH_COMP,
        .computeCodeSize = SHADER_COMMON_MESHFROMDEPTH_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    Recorder recorder(renderer, blurEdgesShader, dataPath, config.targetFramerate);
    Animator animator(animationFile);

    if (saveImage) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();

        animator.copyPoseToCamera(camera);
        animator.copyPoseToCamera(remoteCameraCenter);
    }

    bool generateIFrame = true;
    bool generatePFrame = false;
    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = animationFileIn;
    bool restrictMovementToViewBox = !animationFileIn;
    float viewSphereDiameter = args::get(viewSphereDiameterIn);

    float networkLatency = !animationFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !animationFileIn ? 0.0f : args::get(networkJitterIn);
    PoseSendRecvSimulator poseSendRecvSimulator(networkLatency, networkJitter);
    bool posePrediction = true;
    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !animationFileIn ? 0 : 5; // default to 30fps
    double rerenderInterval = MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    bool* showLayers = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showLayers[i] = true;
    }

    unsigned int totalProxies = 0;
    unsigned int totalDepthOffsets = 0;

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showLayerPreviews = false;
        static bool showCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool showGBufferPreviewWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !animationFileIn ? 0 : 5; // default to 30fps

        static bool showSkyBox = true;

        std::vector<unsigned int> numVertices(maxViews);
        std::vector<unsigned int> numIndicies(maxViews);
        for (int view = 0; view < maxViews; view++) {
            if (!showLayers[view]) {
                continue;
            }

            auto meshBufferSizes = meshFromQuads.getBufferSizes();
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
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 0.1f, 20.0f);

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

            if (ImGui::Checkbox("Correct Normal Orientation", &quadsGenerator.doOrientationCorrection)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Distance Threshold", &quadsGenerator.distanceThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Angle Threshold", &quadsGenerator.angleThreshold, 0.0f, 180.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Flat Threshold (x0.01)", &quadsGenerator.flatThreshold, 0.0f, 10.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.0f, 5.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
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

            if (ImGui::SliderFloat("View Sphere Diameter", &viewSphereDiameter, 0.1f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
                dpRenderer.setViewSphereDiameter(viewSphereDiameter);
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
                    std::vector<Vertex> vertices = meshLayers[view].vertexBuffer.getData();
                    std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), numVertices[view] * sizeof(Vertex));
                    verticesFile.close();
                    spdlog::info("Saved {} vertices ({:.3f} MB) for layer {}", numVertices[view],
                                                (float)numVertices[view] * sizeof(Vertex) / BYTES_IN_MB, view);

                    // save indexBuffer
                    meshLayers[view].indexBuffer.bind();
                    std::vector<unsigned int> indices = meshLayers[view].indexBuffer.getData();
                    std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), numIndicies[view] * sizeof(unsigned int));
                    indicesFile.close();
                    spdlog::info("Saved {} indicies ({:.3f} MB) for layer {}", numIndicies[view],
                                                (float)numIndicies[view] * sizeof(Vertex) / BYTES_IN_MB, view);

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
        dpRenderer.setWindowSize(width, height);
        renderer.setWindowSize(width, height);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    double lastRenderTime = 0.0;
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

        if (animator.running) {
            animator.copyPoseToCamera(camera);
            animator.update(dt);
            if (!animator.running) {
                recorder.stop();
                window->close();

                double avgPosError, avgRotError, avgTimeError, stdPosError, stdRotError, stdTimeError;
                poseSendRecvSimulator.getAvgErrors(avgPosError, avgRotError, avgTimeError, stdPosError, stdRotError, stdTimeError);
                spdlog::info("Pose Error: Pos ({:.2f}±{:.2f}), Rot ({:.2f}±{:.2f}), RTT ({:.2f}±{:.2f})",
                            avgPosError, stdPosError, avgRotError, stdRotError, avgTimeError, stdTimeError);
            }
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

            unsigned int compressedSize = 0;

            totalProxies = 0;
            totalDepthOffsets = 0;

            poseSendRecvSimulator.sendPose(camera, now);
            if (!preventCopyingLocalPose) {
                Pose clientPose;
                if (poseSendRecvSimulator.recvPose(clientPose, now)) {
                    // update center camera
                    remoteCameraCenter.setViewMatrix(clientPose.mono.view);
                    poseSendRecvSimulator.accumulateError(camera, remoteCameraCenter);
                }
                // update wide fov camera
                remoteCameraWideFov.setViewMatrix(remoteCameraCenter.getViewMatrix());
            }

            // render remote scene with multiple layers
            dpRenderer.drawObjects(remoteScene, remoteCameraCenter);
            totalRenderTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
            startTime = window->getTime();

            for (int view = 0; view < maxViews; view++) {
                int hiddenIndex = view - 1;
                auto& remoteCameraToUse = (view == 0 && generatePFrame) ? remoteCameraCenterPrev :
                                            ((view != maxViews - 1) ? remoteCameraCenter : remoteCameraWideFov);

                auto& gBufferToUse = (view == 0) ? gBufferCenterRT : gBufferHiddenRTs[hiddenIndex];
                auto& meshToUse = (view == 0) ? meshesCenter[currMeshIndex] : meshLayers[hiddenIndex];
                auto& currMeshDepth = meshDepths[hiddenIndex];

                int remoteCameraToUseInt = (view == 0 && generatePFrame) ? 0 :
                                            ((view != maxViews - 1) ? 1 : 2);
                if (view == maxViews - 1)
                    spdlog::info("view {} remoteCameraToUseInt {}", view, remoteCameraToUseInt);

                if (view == 0) {
                    remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
                    if (!showNormals) {
                        remoteRenderer.gBuffer.blitToGBuffer(gBufferCenterRT);
                    }
                    else {
                        remoteRenderer.drawToRenderTarget(screenShaderNormals, gBufferCenterRT);
                    }
                }
                else if (view != maxViews - 1) {
                    // copy to render target
                    if (!showNormals) {
                        dpRenderer.peelingLayers[hiddenIndex+1].blitToGBuffer(gBufferToUse);
                    }
                    else {
                        dpRenderer.drawToRenderTarget(screenShaderNormals, gBufferToUse);
                    }
                }
                // wide fov camera
                else {
                    // draw old center mesh at new remoteCamera view, filling stencil buffer with 1
                    wideFOVRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                    wideFOVRenderer.pipeline.writeMaskState.disableColorWrites();
                    nodeMeshes[currMeshIndex].visible = false;
                    nodeMeshes[prevMeshIndex].visible = true;
                    wideFOVRenderer.drawObjectsNoLighting(meshScene, remoteCameraToUse);

                    // render remoteScene using stencil buffer as a mask
                    // at values where stencil buffer is not 1, remoteScene should render
                    wideFOVRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                    wideFOVRenderer.pipeline.writeMaskState.enableColorWrites();
                    wideFOVRenderer.drawObjects(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    wideFOVRenderer.pipeline.stencilState.restoreStencilState();

                    if (!showNormals) {
                        wideFOVRenderer.gBuffer.blitToGBuffer(gBufferToUse);
                    }
                    else {
                        wideFOVRenderer.drawToRenderTarget(screenShaderNormals, gBufferToUse);
                    }
                }

                /*
                ============================
                Generate I-frame
                ============================
                */
                unsigned int numProxies = 0, numDepthOffsets = 0;
                compressedSize += frameGenerator.generateIFrame(
                    gBufferToUse, remoteCameraToUse,
                    quadsGenerator, meshFromQuads, meshToUse,
                    numProxies, numDepthOffsets
                );
                if (!(view == 0 && generatePFrame) && showLayers[view]) {
                    totalProxies += numProxies;
                    totalDepthOffsets += numDepthOffsets;
                }

                totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxies;
                totalCreateMeshTime += frameGenerator.stats.timeToCreateMesh;

                totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuads;
                totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuads;
                totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuads;

                totalAppendProxiesMsTime += frameGenerator.stats.timeToAppendProxies;
                totalFillQuadsIndiciesMsTime += frameGenerator.stats.timeToFillQuadIndices;
                totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertInd;

                /*
                ============================
                Generate P-frame
                ============================
                */
                if (view == 0) {
                    if (generatePFrame) {
                        compressedSize += frameGenerator.generatePFrame(
                            remoteRenderer, remoteScene,
                            meshScenesCenter[currMeshIndex], meshScenesCenter[prevMeshIndex],
                            gBufferCenterTempRT, gBufferCenterMaskRT,
                            remoteCameraCenter, remoteCameraCenterPrev,
                            quadsGenerator, meshFromQuads, meshFromQuadsMask,
                            meshesCenter[currMeshIndex], meshMask,
                            numProxies, numDepthOffsets
                        );
                        totalProxies += numProxies;
                        totalDepthOffsets += numDepthOffsets;

                        totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxies;
                        totalCreateMeshTime += frameGenerator.stats.timeToCreateMesh;

                        totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuads;
                        totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuads;
                        totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuads;

                        totalAppendProxiesMsTime += frameGenerator.stats.timeToAppendProxies;
                        totalFillQuadsIndiciesMsTime += frameGenerator.stats.timeToFillOutputQuads;
                        totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertInd;
                    }
                    nodeMask.visible = generatePFrame;
                    currMeshIndex = (currMeshIndex + 1) % 2;
                    prevMeshIndex = (prevMeshIndex + 1) % 2;

                    // only update the previous camera pose if we are not generating a P-Frame
                    if (!generatePFrame) {
                        remoteCameraCenterPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());
                    }
                }

                if (saveToFile) {
                    unsigned int savedBytes;

                    startTime = window->getTime();
                    std::string quadsFileName = dataPath + "quads" + std::to_string(view) + ".bin";
                    savedBytes = quadsGenerator.saveToFile(quadsFileName);
                    spdlog::info("Saved {} quads ({:.3f} MB) in {:.3f}ms", numProxies,
                                                        (float)savedBytes / BYTES_IN_MB,
                                                        (window->getTime() - startTime) * MILLISECONDS_IN_SECOND);

                    startTime = window->getTime();
                    std::string depthOffsetsFileName = dataPath + "depthOffsets" + std::to_string(view) + ".bin";
                    savedBytes = quadsGenerator.saveDepthOffsetsToFile(depthOffsetsFileName);
                    spdlog::info("Saved {} depth offsets ({:.3f} MB) in {:.3f}ms", numDepthOffsets,
                                                        (float)savedBytes / BYTES_IN_MB,
                                                        (window->getTime() - startTime) * MILLISECONDS_IN_SECOND);

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    gBufferToUse.saveColorAsPNG(colorFileName);
                }

                // For debugging: Generate point cloud from depth map
                if (showDepth) {
                    glm::vec2 gBufferSize = glm::vec2(gBufferToUse.width, gBufferToUse.height);

                    meshFromDepthShader.startTiming();

                    meshFromDepthShader.bind();
                    {
                        meshFromDepthShader.setTexture(gBufferToUse.depthStencilBuffer, 0);
                    }
                    {
                        meshFromDepthShader.setVec2("depthMapSize", gBufferSize);
                    }
                    {
                        meshFromDepthShader.setMat4("view", remoteCameraCenter.getViewMatrix());
                        meshFromDepthShader.setMat4("projection", remoteCameraCenter.getProjectionMatrix());
                        meshFromDepthShader.setMat4("viewInverse", remoteCameraCenter.getViewMatrixInverse());
                        meshFromDepthShader.setMat4("projectionInverse", remoteCameraCenter.getProjectionMatrixInverse());

                        meshFromDepthShader.setFloat("near", remoteCameraCenter.getNear());
                        meshFromDepthShader.setFloat("far", remoteCameraCenter.getFar());
                    }
                    {
                        meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currMeshDepth.vertexBuffer);
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
            spdlog::info("Frame Size: {:.3f}MB", (float)(compressedSize) / BYTES_IN_MB);

            preventCopyingLocalPose = false;
            generateIFrame = false;
            generatePFrame = false;
            saveToFile = false;
        }

        // hide/show nodeLayers based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showLayer = showLayers[view];

            if (view == 0) {
                // show previous mesh
                nodeMeshesLocal[currMeshIndex].visible = false;
                nodeMeshesLocal[prevMeshIndex].visible = showLayer;
                nodeWireframesCenter[currMeshIndex].visible = false;
                nodeWireframesCenter[prevMeshIndex].visible = showLayer && showWireframe;
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
        blurEdgesShader.bind();
        blurEdgesShader.setBool("toneMap", !showNormals);
        renderer.drawToScreen(blurEdgesShader);
        if (animator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", (window->getTime() - startTime) * MILLISECONDS_IN_SECOND);
        }

        if (animator.running || recording) {
            recorder.captureFrame(camera);
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
