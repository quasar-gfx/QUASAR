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

#include <Recorder.h>
#include <CameraAnimator.h>

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
    glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
    glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "QuadStream Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::ValueFlag<float> viewBoxSizeIn(parser, "view-box-size", "Size of view box in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 60.0f);
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
    glm::uvec2 halfRemoteWindowSize = remoteWindowSize / 2u;

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
    PerspectiveCamera remoteCamera(remoteRenderer.width, remoteRenderer.height);
    PerspectiveCamera remoteCameraPrev(remoteRenderer.width, remoteRenderer.height);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    // "local" scene
    Scene localScene;
    localScene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    // scenes with resulting mesh
    std::vector<Scene> meshScenes(2);
    int currMeshIndex = 0, prevMeshIndex = 1;

    QuadsGenerator quadsGenerator(halfRemoteWindowSize);
    MeshFromQuads meshFromQuads(halfRemoteWindowSize);
    FrameGenerator frameGenerator(remoteRenderer, remoteScene, quadsGenerator, meshFromQuads);

    unsigned int maxVertices = MAX_NUM_PROXIES * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

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
    GBuffer gBufferRT(rtParams);
    GBuffer gBufferMaskRT(rtParams);
    GBuffer gBufferTemp(rtParams);

    rtParams.width = halfRemoteWindowSize.x;
    rtParams.height = halfRemoteWindowSize.y;
    GBuffer gBufferRTLowRes(rtParams);
    GBuffer gBufferMaskRTLowRes(rtParams);
    GBuffer gBufferTempRTLowRes(rtParams);

    std::vector<Mesh> meshes; meshes.reserve(2);
    std::vector<Node> nodeMeshes; nodeMeshes.reserve(2);
    std::vector<Node> nodeMeshesLocal; nodeMeshesLocal.reserve(2);
    std::vector<Node> nodeWireframes; nodeWireframes.reserve(2);

    MeshSizeCreateParams meshParams = {
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = new QuadMaterial({ .baseColorTexture = &gBufferRT.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    };
    for (int i = 0; i < 2; i++) {
        meshes.emplace_back(meshParams);

        nodeMeshes.emplace_back(&meshes[i]);
        nodeMeshes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&nodeMeshes[i]);

        nodeMeshesLocal.emplace_back(&meshes[i]);
        nodeMeshesLocal[i].frustumCulled = false;
        localScene.addChildNode(&nodeMeshesLocal[i]);

        nodeWireframes.emplace_back(&meshes[i]);
        nodeWireframes[i].frustumCulled = false;
        nodeWireframes[i].wireframe = true;
        nodeWireframes[i].visible = false;
        nodeWireframes[i].overrideMaterial = new QuadMaterial({ .baseColor = colors[0] });
        localScene.addChildNode(&nodeWireframes[i]);
    }

    meshParams.material = new QuadMaterial({ .baseColorTexture = &gBufferMaskRT.colorBuffer });
    Mesh meshMask(meshParams);
    Node nodeMask(&meshMask);
    nodeMask.frustumCulled = false;

    Node nodeMaskWireframe(&meshMask);
    nodeMaskWireframe.frustumCulled = false;
    nodeMaskWireframe.wireframe = true;
    nodeMaskWireframe.visible = false;
    nodeMaskWireframe.overrideMaterial = new UnlitMaterial({ .baseColor = colors[colors.size()-1] });

    localScene.addChildNode(&nodeMask);
    localScene.addChildNode(&nodeMaskWireframe);

    Mesh meshDepth = Mesh({
        .maxVertices = maxVerticesDepth,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDepth = Node(&meshDepth);
    nodeDepth.frustumCulled = false;
    nodeDepth.visible = false;
    nodeDepth.primativeType = GL_POINTS;
    localScene.addChildNode(&nodeDepth);

    // shaders
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

    // post processing
    ToneMapper toneMapper;

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

    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewBox = !cameraPathFileIn;
    float viewBoxSize = args::get(viewBoxSizeIn);

    bool generateIFrame = true;
    bool generatePFrame = false;

    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps
    double rerenderInterval = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = true;
    PoseSendRecvSimulator poseSendRecvSimulator(networkLatency, networkJitter, rerenderInterval / MILLISECONDS_IN_SECOND);

    unsigned int totalProxies = 0;
    unsigned int totalDepthOffsets = 0;

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = !saveImages;
        static bool showCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool showGBufferPreviewWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30fps

        static bool showSkyBox = true;

        auto meshBufferSizes = frameGenerator.meshFromQuads.getBufferSizes();
        auto meshBufferSizesMask = frameGenerator.meshFromQuadsMask.getBufferSizes();

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

            unsigned int totalTriangles = (meshBufferSizes.numIndices + meshBufferSizesMask.numIndices) / 3;
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

                if (ImGui::DragFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.001f, 0.0f, 1.0f)) {
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

            if (ImGui::DragFloat("View Box Size", &viewBoxSize, 0.05f, 0.1f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

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

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            std::string colorFileName = dataPath + "color.png";

            if (ImGui::Button("Save Mesh")) {
                std::string verticesFileName = dataPath + "vertices.bin";
                std::string indicesFileName = dataPath + "indices.bin";

                // save vertexBuffer
                meshes[currMeshIndex].vertexBuffer.bind();
                std::vector<QuadVertex> vertices = meshes[currMeshIndex].vertexBuffer.getData<QuadVertex>();
                std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                verticesFile.write((char*)vertices.data(), meshBufferSizes.numVertices * sizeof(QuadVertex));
                verticesFile.close();
                spdlog::info("Saved {} vertices ({:.3f} MB)", meshBufferSizes.numVertices,
                                        (float)meshBufferSizes.numVertices * sizeof(QuadVertex) / BYTES_IN_MB);

                // save indexBuffer
                meshes[currMeshIndex].indexBuffer.bind();
                std::vector<unsigned int> indices = meshes[currMeshIndex].indexBuffer.getData<unsigned int>();
                std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                indicesFile.write((char*)indices.data(), meshBufferSizes.numIndices * sizeof(unsigned int));
                indicesFile.close();
                spdlog::info("Saved {} indices ({:.3f} MB)", meshBufferSizes.numIndices,
                                        (float)meshBufferSizes.numIndices * sizeof(unsigned int) / BYTES_IN_MB);

                // save color buffer
                gBufferRT.saveColorAsPNG(colorFileName);
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
            ImGui::Image((void*)(intptr_t)(gBufferRT.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::Begin("GBuffer Mask Color", 0, flags);
            ImGui::Image((void*)(intptr_t)(gBufferMaskRT.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
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
                    remoteCamera.setViewMatrix(clientPosePred.mono.view);
                }
                // if we do not have a new pose, just send a new frame with the old pose
            }

            auto& remoteCameraToUse = generatePFrame ? remoteCameraPrev : remoteCamera;

            // render all objects in remoteScene normally
            remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
            if (!showNormals) {
                remoteRenderer.copyToGBuffer(gBufferRTLowRes);
                remoteRenderer.copyToGBuffer(gBufferRT);
            }
            else {
                remoteRenderer.drawToRenderTarget(screenShaderNormals, gBufferRTLowRes);
                remoteRenderer.drawToRenderTarget(screenShaderNormals, gBufferRT);
            }
            totalRenderTime += timeutils::secondsToMillis(window->getTime() - startTime);

            /*
            ============================
            Generate I-frame
            ============================
            */
            unsigned int numProxies = 0, numDepthOffsets = 0;
            quadsGenerator.expandEdges = false;
            compressedSize = frameGenerator.generateIFrame(
                gBufferRTLowRes, gBufferRT,
                remoteCameraToUse,
                meshes[currMeshIndex],
                numProxies, numDepthOffsets
            );
            if (!generatePFrame) {
                totalProxies += numProxies;
                totalDepthOffsets += numDepthOffsets;

                totalCompressTime += frameGenerator.stats.timeToCompress;
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
            if (generatePFrame) {
                quadsGenerator.expandEdges = true;
                compressedSize = frameGenerator.generatePFrame(
                    meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                    gBufferTemp, gBufferMaskRT,
                    gBufferTempRTLowRes, gBufferMaskRTLowRes,
                    remoteCamera, remoteCameraPrev,
                    meshes[currMeshIndex], meshMask,
                    numProxies, numDepthOffsets
                );
                totalProxies += numProxies;
                totalDepthOffsets += numDepthOffsets;

                totalCompressTime += frameGenerator.stats.timeToCompress;

                totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
                totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

                totalAppendProxiesMsTime += frameGenerator.stats.timeToAppendProxiesMs;
                totalFillQuadsIndiciesMsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
                totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;
            }
            nodeMask.visible = generatePFrame;
            currMeshIndex = (currMeshIndex + 1) % 2;
            prevMeshIndex = (prevMeshIndex + 1) % 2;

            // only update the previous camera pose if we are not generating a P-Frame
            if (generateIFrame) {
                remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
            }

            // save to file if requested
            if (saveToFile) {
                unsigned int savedBytes;

                startTime = window->getTime();
                savedBytes = quadsGenerator.saveToFile(dataPath + "quads.bin");
                spdlog::info("Saved {} quads ({:.3f} MB) in {:.3f}ms", numProxies,
                                                    (float)savedBytes / BYTES_IN_MB,
                                                    timeutils::secondsToMillis(window->getTime() - startTime));

                startTime = window->getTime();
                savedBytes = quadsGenerator.saveDepthOffsetsToFile(dataPath + "depthOffsets.bin");
                spdlog::info("Saved {} depth offsets ({:.3f} MB) in {:.3f}ms", numDepthOffsets,
                                                    (float)savedBytes / BYTES_IN_MB,
                                                    timeutils::secondsToMillis(window->getTime() - startTime));

                // save color buffer
                std::string colorFileName = dataPath + "color.png";
                gBufferRT.saveColorAsPNG(colorFileName);
            }

            // For debugging: Generate point cloud from depth map
            if (showDepth) {
                const glm::vec2 gBufferSize = glm::vec2(gBufferRTLowRes.width, gBufferRTLowRes.height);

                meshFromDepthShader.startTiming();

                meshFromDepthShader.bind();
                {
                    meshFromDepthShader.setTexture(gBufferRTLowRes.depthStencilBuffer, 0);
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
                    meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshDepth.vertexBuffer);
                    meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
                }
                meshFromDepthShader.dispatch((gBufferSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                             (gBufferSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                meshFromDepthShader.endTiming();
                totalGenDepthTime += meshFromDepthShader.getElapsedTime();
            }

            spdlog::info("======================================================");
            spdlog::info("Rendering Time: {:.3f}ms", totalRenderTime);
            if (generatePFrame) spdlog::info("Time To Render Masks Time: {:.3f}ms", frameGenerator.stats.timeToRenderMasks);
            spdlog::info("Create Proxies Time: {:.3f}ms", totalCreateProxiesTime);
            spdlog::info("  Gen Quad Map Time: {:.3f}ms", totalGenQuadMapTime);
            spdlog::info("  Simplify Time: {:.3f}ms", totalSimplifyTime);
            spdlog::info("  Fill Quads Time: {:.3f}ms", totalFillQuadsTime);
            spdlog::info("Create Mesh Time: {:.3f}ms", totalCreateMeshTime);
            spdlog::info("  Append Quads Time: {:.3f}ms", totalAppendProxiesMsTime);
            spdlog::info("  Fill Output Quads Time: {:.3f}ms", totalFillQuadsIndiciesMsTime);
            spdlog::info("  Create Vert/Ind Time: {:.3f}ms", totalCreateVertIndTime);
            spdlog::info("Compress Time: {:.3f}ms", totalCompressTime);
            if (showDepth) spdlog::info("Gen Depth Time: {:.3f}ms", totalGenDepthTime);
            spdlog::info("Frame Size: {:.3f}MB", static_cast<float>(compressedSize) / BYTES_IN_MB);

            preventCopyingLocalPose = false;
            generateIFrame = false;
            generatePFrame = false;
            saveToFile = false;
        }

        poseSendRecvSimulator.update(now);

        if (!updateClient) {
            return;
        }

        // show previous mesh
        nodeMeshesLocal[currMeshIndex].visible = false;
        nodeMeshesLocal[prevMeshIndex].visible = true;
        nodeWireframes[currMeshIndex].visible = false;
        nodeWireframes[prevMeshIndex].visible = showWireframe;

        nodeMaskWireframe.visible = nodeMask.visible && showWireframe;
        nodeDepth.visible = showDepth;

        if (restrictMovementToViewBox) {
            glm::vec3 remotePosition = remoteCamera.getPosition();
            glm::vec3 position = camera.getPosition();
            // restrict camera position to be inside position±viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - viewBoxSize/2, remotePosition.x + viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewBoxSize/2, remotePosition.y + viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewBoxSize/2, remotePosition.z + viewBoxSize/2);
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        double startTime = window->getTime();

        // render generated meshes
        renderStats = renderer.drawObjects(localScene, camera);

        // render to screen
        toneMapper.enableToneMapping(!showNormals);
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
