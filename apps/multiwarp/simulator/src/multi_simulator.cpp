#include <filesystem>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <Shaders/ToneMapShader.h>

#include <Recorder.h>
#include <Animator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <PoseSendRecvSimulator.h>

#include <shaders_common.h>

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

int main(int argc, char** argv) {
    Config config{};
    config.title = "Multi-Camera Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'V', "vsync"}, true);
    args::Flag saveImage(parser, "save", "Save outputs to disk", {'I', "save-image"});
    args::ValueFlag<std::string> animationFileIn(parser, "anim-path", "Path to camera animation file", {'A', "animation-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
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

    // 0th is standard view, maxViews-1 is large fov view
    int maxAdditionalViews = args::get(maxAdditionalViewsIn);
    int maxViews = maxAdditionalViews + 2;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    ForwardRenderer remoteRenderer(config);
    ForwardRenderer wideFOVRenderer(config);

    // "remote" scene
    Scene remoteScene;
    std::vector<PerspectiveCamera> remoteCameras; remoteCameras.reserve(maxViews);
    for (int view = 0; view < maxViews; view++) {
        remoteCameras.emplace_back(remoteRenderer.width, remoteRenderer.height);
    }
    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCameraCenter);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCameraCenter.setFovyDegrees(remoteFOV);

     // make last camera have a larger fov
    float remoteFOVWide = args::get(remoteFOVWideIn);
    remoteCameras[maxViews-1].setFovyDegrees(remoteFOVWide);

    // scene with all the meshViews
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCameraCenter.getViewMatrix());

    FrameGenerator frameGenerator;
    QuadsGenerator quadsGenerator(remoteWindowSize);
    quadsGenerator.expandEdges = true;
    quadsGenerator.depthThreshold = 1e-4f;
    quadsGenerator.flatThreshold = 1e-2f;
    quadsGenerator.proxySimilarityThreshold = 0.1f;
    MeshFromQuads meshFromQuads(remoteWindowSize);

    std::vector<GBuffer> gBufferRTs; gBufferRTs.reserve(maxViews);
    RenderTargetCreateParams params = {
        .width = windowSize.x,
        .height = windowSize.y,
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
            params.width = 1280; params.height = 720;
        }
        gBufferRTs.emplace_back(params);
    }

    unsigned int maxVertices = MAX_NUM_PROXIES * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    Scene meshScene;
    std::vector<Mesh> meshViews; meshViews.reserve(maxViews);
    std::vector<Mesh> meshDepths; meshDepths.reserve(maxViews);

    std::vector<Node> nodes; nodes.reserve(maxViews);
    std::vector<Node> nodeMeshes; nodeMeshes.reserve(maxViews);
    std::vector<Node> nodeWireframes; nodeWireframes.reserve(maxViews);
    std::vector<Node> nodeDepths; nodeDepths.reserve(maxViews);

    for (int view = 0; view < maxViews; view++) {
        MeshSizeCreateParams meshParams = {
            .maxVertices = maxVertices / (view == 0 || view == maxViews - 1 ? 1 : 2),
            .maxIndices = maxIndices / (view == 0 || view == maxViews - 1 ? 1 : 2),
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .material = new QuadMaterial({ .baseColorTexture = &gBufferRTs[view].colorBuffer }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        };
        meshViews.emplace_back(meshParams);

        nodes.emplace_back(&meshViews[view]);
        nodes[view].frustumCulled = false;
        scene.addChildNode(&nodes[view]);

        const glm::vec4 &color = colors[view % colors.size()];

        nodeWireframes.emplace_back(&meshViews[view]);
        nodeWireframes[view].frustumCulled = false;
        nodeWireframes[view].wireframe = true;
        nodeWireframes[view].overrideMaterial = new UnlitMaterial({ .baseColor = color });
        scene.addChildNode(&nodeWireframes[view]);

        MeshSizeCreateParams meshDepthParams = {
            .maxVertices = maxVerticesDepth,
            .material = new UnlitMaterial({ .baseColor = color }),
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
    ToneMapShader toneMapShader;

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

    Recorder recorder(renderer, toneMapShader, dataPath, config.targetFramerate);
    Animator animator(animationFile);

    if (saveImage) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();

        animator.copyPoseToCamera(camera);
        animator.copyPoseToCamera(remoteCameraCenter);
    }

    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = animationFileIn;
    bool restrictMovementToViewBox = !animationFileIn;
    float viewBoxSize = args::get(viewBoxSizeIn);

    bool rerender = true;

    float networkLatency = !animationFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !animationFileIn ? 0.0f : args::get(networkJitterIn);
    PoseSendRecvSimulator poseSendRecvSimulator(networkLatency, networkJitter);
    bool posePrediction = true;
    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !animationFileIn ? 0 : 5; // default to 30fps
    double rerenderInterval = MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    bool* showViews = new bool[maxViews];
    for (int view = 0; view < maxViews; ++view) {
        showViews[view] = true;
    }

    unsigned int totalProxies = 0;
    unsigned int totalDepthOffsets = 0;

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showViewPreviews = false;
        static bool showCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !animationFileIn ? 0 : 5; // default to 30fps

        static bool showSkyBox = true;

        std::vector<unsigned int> numVertices(maxViews);
        std::vector<unsigned int> numIndicies(maxViews);
        for (int view = 0; view < maxViews; view++) {
            if (!showViews[view]) {
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

            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::Checkbox("Correct Normal Orientation", &quadsGenerator.correctOrientation)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Depth Threshold", &quadsGenerator.depthThreshold, 0.0f, 1.0f, "%.4f")) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Angle Threshold", &quadsGenerator.angleThreshold, 0.0f, 180.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Flat Threshold", &quadsGenerator.flatThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.0f, 1.0f)) {
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

            ImGui::Separator();

            if (ImGui::SliderFloat("View Box Size", &viewBoxSize, 0.1f, 1.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
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
            std::string fileName = dataPath + std::string(fileNameBase) + "." + time;

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);

                for (int view = 1; view < maxViews; view++) {
                    fileName = dataPath + std::string(fileNameBase) + ".view" + std::to_string(view) + "." + time;
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

            if (ImGui::Button("Save Mesh")) {
                for (int view = 0; view < maxViews; view++) {
                    std::string verticesFileName = dataPath + "vertices" + std::to_string(view) + ".bin";
                    std::string indicesFileName = dataPath + "indices" + std::to_string(view) + ".bin";

                    // save vertexBuffer
                    meshViews[view].vertexBuffer.bind();
                    std::vector<QuadVertex> vertices = meshViews[view].vertexBuffer.getData<QuadVertex>();
                    std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), numVertices[view] * sizeof(QuadVertex));
                    verticesFile.close();
                    spdlog::info("Saved {} vertices ({:.3f} MB) for view {}", numVertices[view],
                                                (float)numVertices[view] * sizeof(QuadVertex) / BYTES_IN_MB, view);

                    // save indexBuffer
                    meshViews[view].indexBuffer.bind();
                    std::vector<unsigned int> indices = meshViews[view].indexBuffer.getData<unsigned int>();
                    std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), numIndicies[view] * sizeof(unsigned int));
                    indicesFile.close();
                    spdlog::info("Saved {} indicies ({:.3f} MB) for view {}", numIndicies[view],
                                                (float)numIndicies[view] * sizeof(unsigned int) / BYTES_IN_MB, view);

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    gBufferRTs[view].saveColorAsPNG(colorFileName);
                }
            }

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
                saveToFile = true;
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

                // update other cameras in view box corners
                for (int view = 1; view < maxViews - 1; view++) {
                    const glm::vec3 &offset = offsets[view - 1];
                    remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
                    remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + viewBoxSize/2 * offset);
                    remoteCameras[view].updateViewMatrix();
                }

                // update wide fov camera
                remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());
            }

            for (int view = 0; view < maxViews; view++) {
                auto& remoteCamera = remoteCameras[view];

                auto& currMesh = meshViews[view];
                auto& currMeshDepth = meshDepths[view];

                startTime = window->getTime();

                auto &renderer = (view != maxViews - 1) ? remoteRenderer : wideFOVRenderer;

                // center view
                if (view == 0) {
                    // render all objects in remoteScene normally
                    renderer.drawObjects(remoteScene, remoteCamera);
                }
                // other views
                else {
                    // make all previous meshViews visible and everything else invisible
                    for (int prevView = 1; prevView < maxViews; prevView++) {
                        meshScene.rootNode.children[prevView]->visible = (prevView < view);
                    }
                    // draw old meshViews at new remoteCamera view, filling stencil buffer with 1
                    renderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                    renderer.pipeline.writeMaskState.disableColorWrites();
                    renderer.drawObjectsNoLighting(meshScene, remoteCamera);

                    // render remoteScene using stencil buffer as a mask
                    // at values where stencil buffer is not 1, remoteScene should render
                    renderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                    renderer.pipeline.rasterState.polygonOffsetEnabled = false;
                    renderer.pipeline.writeMaskState.enableColorWrites();
                    renderer.drawObjects(remoteScene, remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    renderer.pipeline.stencilState.restoreStencilState();
                }
                if (!showNormals) {
                    renderer.gBuffer.blitToGBuffer(gBufferRTs[view]);
                }
                else {
                    renderer.drawToRenderTarget(screenShaderNormals, gBufferRTs[view]);
                }
                totalRenderTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;

                unsigned int numProxies = 0, numDepthOffsets = 0;
                compressedSize += frameGenerator.generateIFrame(
                    gBufferRTs[view], gBufferRTs[view], remoteCamera,
                    quadsGenerator, meshFromQuads, currMesh,
                    numProxies, numDepthOffsets,
                    false
                );
                if (showViews[view]) {
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
                    gBufferRTs[view].saveColorAsPNG(colorFileName);
                }

                // For debugging: Generate point cloud from depth map
                if (showDepth) {
                    const glm::vec2 gBufferSize = glm::vec2(gBufferRTs[view].width, gBufferRTs[view].height);

                    meshFromDepthShader.startTiming();

                    meshFromDepthShader.bind();
                    {
                        meshFromDepthShader.setTexture(gBufferRTs[view].depthStencilBuffer, 0);
                    }
                    {
                        meshFromDepthShader.setVec2("depthMapSize", gBufferSize);
                    }
                    {
                        meshFromDepthShader.setMat4("view", remoteCamera.getViewMatrix());
                        meshFromDepthShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                        meshFromDepthShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
                        meshFromDepthShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());

                        meshFromDepthShader.setFloat("near", remoteCamera.getNear());
                        meshFromDepthShader.setFloat("far", remoteCamera.getFar());
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
            // QS has data structures that are 103 bits
            spdlog::info("Frame Size: {:.3f}MB", (float)(compressedSize) / BYTES_IN_MB * (103.0) / (8*sizeof(QuadMapDataPacked)));
            spdlog::info("Num Proxies: {}Proxies", totalProxies);

            preventCopyingLocalPose = false;
            rerender = false;
            saveToFile = false;
        }

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
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", !showNormals);
        renderer.drawToScreen(toneMapShader);
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
