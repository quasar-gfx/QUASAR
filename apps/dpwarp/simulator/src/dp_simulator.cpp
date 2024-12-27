#include <iostream>
#include <filesystem>
#include <queue>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Recorder.h>
#include <Animator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <shaders_common.h>

int main(int argc, char** argv) {
    spdlog::set_pattern("[%H:%M:%S] [%^%L%$] %v");

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
    args::PositionalList<float> poseOffset(parser, "pose-offset", "Offset for the pose (only used when --save-image is set)");
    args::ValueFlag<int> maxLayersIn(parser, "layers", "Max layers", {'n', "max-layers"}, 4);
    args::Flag disableWideFov(parser, "disable-wide-fov", "Disable wide fov view", {'W', "disable-wide-fov"});
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
    int maxViews = !disableWideFov ? maxLayers + 1 : maxLayers;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    ForwardRenderer remoteRenderer(config);
    DepthPeelingRenderer dpRenderer(config, maxLayers, true);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCameraCenter(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCameraCenter);

    PerspectiveCamera remoteCameraWideFov(remoteWindowSize.x, remoteWindowSize.y);
    if (!disableWideFov) {
        // make last camera have a larger fov
        remoteCameraWideFov.setFovyDegrees(120.0f);
    }

    // scene with all the meshes
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCameraCenter.getViewMatrix());

    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);

    std::vector<RenderTarget> renderTargets; renderTargets.reserve(maxViews);
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
        renderTargets.emplace_back(params);
    }

    unsigned int maxVertices = MAX_NUM_PROXIES * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

    std::vector<Mesh*> meshDepths(maxViews);
    std::vector<Node*> nodeDepths(maxViews);

    for (int view = 0; view < maxViews; view++) {
        meshes[view] = new Mesh({
            .numVertices = maxVertices / (view == 0 || (!disableWideFov && view == maxViews - 1) ? 1 : 4),
            .numIndices = maxIndices / (view == 0 || (!disableWideFov && view == maxViews - 1) ? 1 : 4),
            .material = new QuadMaterial({ .baseColorTexture = &renderTargets[view].colorBuffer }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        });
        nodes[view] = new Node(meshes[view]);
        nodes[view]->frustumCulled = false;
        scene.addChildNode(nodes[view]);

        // primary view color is yellow
        glm::vec4 color = (view == 0) ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) :
                  glm::vec4(fmod(view * 0.6180339887f, 1.0f),
                            fmod(view * 0.9f, 1.0f),
                            fmod(view * 0.5f, 1.0f),
                            1.0f);

        nodeWireframes[view] = new Node(meshes[view]);
        nodeWireframes[view]->frustumCulled = false;
        nodeWireframes[view]->wireframe = true;
        nodeWireframes[view]->overrideMaterial = new UnlitMaterial({ .baseColor = color });
        scene.addChildNode(nodeWireframes[view]);

        meshDepths[view] = new Mesh({
            .numVertices = maxVerticesDepth,
            .material = new UnlitMaterial({ .baseColor = color }),
            .usage = GL_DYNAMIC_DRAW
        });
        nodeDepths[view] = new Node(meshDepths[view]);
        nodeDepths[view]->frustumCulled = false;
        nodeDepths[view]->primativeType = GL_POINTS;
        scene.addChildNode(nodeDepths[view]);
    }

    Scene meshScene;
    Node* node = new Node(meshes[0]);
    node->frustumCulled = false;
    meshScene.addChildNode(node);

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

    bool rerender = true;
    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = animationFileIn;
    bool restrictMovementToViewBox = !animationFileIn;
    float viewSphereDiameter = 0.5f;

    double rerenderInterval = 0.0;
    float networkLatency = !animationFileIn ? 0.0 : 25.0;
    std::queue<Animator::CameraPose> cameraPoses;
    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};

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
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !animationFileIn ? 0 : 5;

        static bool showEnvMap = true;

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

            if (ImGui::Checkbox("Show Environment Map", &showEnvMap)) {
                scene.envCubeMap = showEnvMap ? remoteScene.envCubeMap : nullptr;
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

            if (ImGui::Checkbox("Correct Normal Orientation", &quadsGenerator.doOrientationCorrection)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Distance Threshold", &quadsGenerator.distanceThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Angle Threshold", &quadsGenerator.angleThreshold, 0.0f, 180.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Flat Threshold (x0.01)", &quadsGenerator.flatThreshold, 0.0f, 10.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            if (ImGui::SliderFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.0f, 5.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
            }

            ImGui::Separator();

            if (ImGui::SliderFloat("View Sphere Diameter", &viewSphereDiameter, 0.1f, 5.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
                runAnimations = false;
                dpRenderer.setViewSphereDiameter(viewSphereDiameter);
            }

            ImGui::Checkbox("Restrict Movement to View Sphere", &restrictMovementToViewBox);

            ImGui::Separator();

            ImGui::SliderFloat("Network Latency (ms)", &networkLatency, 0.0f, 1000.0f);

            ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels));
            rerenderInterval = 1000.0 / serverFPSValues[serverFPSIndex];

            if (ImGui::Button("Send Server Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                rerender = true;
                runAnimations = true;
            }

            ImGui::Separator();

            const int columns = 3;
            for (int i = 0; i < maxViews; i++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(i)).c_str(), &showLayers[i]);
                if ((i + 1) % columns != 0) {
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
                    ImGui::Image((void*)(intptr_t)(renderTargets[viewIdx].colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
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
                        renderTargets[view].saveColorAsHDR(fileName + ".hdr");
                    }
                    else {
                        renderTargets[view].saveColorAsPNG(fileName + ".png");
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
                    meshes[view]->vertexBuffer.bind();
                    std::vector<Vertex> vertices = meshes[view]->vertexBuffer.getData();
                    std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), numVertices[view] * sizeof(Vertex));
                    verticesFile.close();
                    spdlog::info("Saved {} vertices ({:.3f} MB) for layer {}", numVertices[view],
                                                (float)numVertices[view] * sizeof(Vertex) / BYTES_IN_MB, view);

                    // save indexBuffer
                    meshes[view]->indexBuffer.bind();
                    std::vector<unsigned int> indices = meshes[view]->indexBuffer.getData();
                    std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), numIndicies[view] * sizeof(unsigned int));
                    indicesFile.close();
                    spdlog::info("Saved {} indicies ({:.3f} MB) for layer {}", numIndicies[view],
                                                (float)numIndicies[view] * sizeof(Vertex) / BYTES_IN_MB, view);

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    renderTargets[view].saveColorAsPNG(colorFileName);
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
        dpRenderer.setWindowSize(width, height);
        renderer.setWindowSize(width, height);

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
            if (!animator.running) {
                window->close();
            }
        }
        else {
            // handle keyboard input
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

        if (rerenderInterval > 0 && now - lastRenderTime > rerenderInterval / MILLISECONDS_IN_SECOND) {
            rerender = true;
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
            double totalGenDepthTime = 0.0;

            totalProxies = 0;
            totalDepthOffsets = 0;

            cameraPoses.push({camera.getPosition(), camera.getRotationQuat(), now});
            if (!preventCopyingLocalPose) {
                while (!cameraPoses.empty() && now - cameraPoses.front().timestamp >= networkLatency / MILLISECONDS_IN_SECOND) {
                    auto pose = cameraPoses.front();

                    // update center camera
                    remoteCameraCenter.setPosition(pose.position);
                    remoteCameraCenter.setRotationQuat(pose.rotation);
                    remoteCameraCenter.updateViewMatrix();

                    // update wide fov camera
                    remoteCameraWideFov.setViewMatrix(remoteCameraCenter.getViewMatrix());

                    cameraPoses.pop();
                }
            }

            /*
            ============================
            FIRST PASS: Render the scene to a G-Buffer render target
            ============================
            */
            dpRenderer.drawObjects(remoteScene, remoteCameraCenter);
            totalRenderTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
            startTime = window->getTime();

            for (int view = 0; view < maxViews; view++) {
                auto& remoteCamera = (view < maxViews - 1) ? remoteCameraCenter : remoteCameraWideFov;

                auto* currMesh = meshes[view];
                auto* currMeshDepth = meshDepths[view];

                /*
                ============================
                FIRST PASS: Render the scene to a G-Buffer render target
                ============================
                */
                if (disableWideFov || view < maxViews - 1) {
                    // render to render target
                    if (!showNormals) {
                        dpRenderer.peelingLayers[view]->blitToRenderTarget(renderTargets[view]);
                    }
                    else {
                        dpRenderer.drawToRenderTarget(screenShaderNormals, renderTargets[view]);
                    }
                }
                // wide fov camera
                else {
                    // draw old meshes at new remoteCamera view, filling depth buffer
                    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                    remoteRenderer.drawObjectsNoLighting(meshScene, remoteCamera);

                    // render remoteScene into stencil buffer, with depth buffer from meshScene
                    // this should draw objects in remoteScene that are not occluded by meshScene, setting
                    // the stencil buffer to 1 where the depth of remoteScene is less than meshScene
                    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();
                    remoteRenderer.pipeline.rasterState.polygonOffsetEnabled = true;
                    remoteRenderer.pipeline.rasterState.polygonOffsetUnits = 10000.0f;
                    remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCamera, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                    // render remoteScene using stencil buffer as a mask
                    // at values were stencil buffer is 1, remoteScene should render
                    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_EQUAL, 1);
                    remoteRenderer.pipeline.rasterState.polygonOffsetEnabled = false;
                    remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                    remoteRenderer.drawObjects(remoteScene, remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    remoteRenderer.pipeline.stencilState.restoreStencilState();

                    if (!showNormals) {
                        toneMapShader.bind();
                        toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                        remoteRenderer.drawToRenderTarget(toneMapShader, renderTargets[view]);
                    }
                    else {
                        remoteRenderer.drawToRenderTarget(screenShaderNormals, renderTargets[view]);
                    }
                }

                // create proxies from the new frame
                startTime = window->getTime();
                auto* gBuffer = (disableWideFov || view != maxViews - 1) ? dpRenderer.peelingLayers[view] : &remoteRenderer.gBuffer;
                auto sizes = quadsGenerator.createProxiesFromGBuffer(*gBuffer, remoteCamera);
                unsigned int numProxies = sizes.numProxies;
                unsigned int numDepthOffsets = sizes.numDepthOffsets;
                totalProxies += numProxies;
                totalDepthOffsets += numDepthOffsets;
                totalCreateProxiesTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
                totalGenQuadMapTime += quadsGenerator.stats.timeToGenerateQuadsMs;
                totalSimplifyTime += quadsGenerator.stats.timeToSimplifyQuadsMs;
                totalFillQuadsTime += quadsGenerator.stats.timeToFillOutputQuadsMs;

                if (saveToFile) {
                    startTime = window->getTime();

                    unsigned int savedBytes;

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
                    renderTargets[view].saveColorAsPNG(colorFileName);
                }

                // create mesh from proxies
                startTime = window->getTime();
                meshFromQuads.appendProxies(numProxies, quadsGenerator.outputQuadBuffers);
                meshFromQuads.createMeshFromProxies(
                    numProxies, quadsGenerator.depthOffsets,
                    remoteCamera,
                    *currMesh
                );
                totalCreateMeshTime += meshFromQuads.stats.timeToCreateMeshMs;

                // For debugging: Generate point cloud from depth map
                if (showDepth) {
                    meshFromDepthShader.startTiming();

                    meshFromDepthShader.bind();
                    {
                        if (disableWideFov || view != maxViews - 1) {
                            meshFromDepthShader.setTexture(dpRenderer.peelingLayers[view]->depthStencilBuffer, 0);
                        }
                        else {
                            meshFromDepthShader.setTexture(remoteRenderer.gBuffer.depthStencilBuffer, 0);
                        }
                    }
                    {
                        meshFromDepthShader.setVec2("depthMapSize", remoteWindowSize);
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
                        meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currMeshDepth->vertexBuffer);
                        meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
                    }
                    meshFromDepthShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                 (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
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
            if (showDepth) spdlog::info("Gen Depth Time: {:.3f}ms", totalGenDepthTime);

            preventCopyingLocalPose = false;
            rerender = false;
            saveToFile = false;
        }

        // hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showLayer = showLayers[view];

            nodes[view]->visible = showLayer;
            nodeWireframes[view]->visible = showLayer && showWireframe;
            nodeDepths[view]->visible = showLayer && showDepth;
        }

        if (saveImage && args::get(poseOffset).size() == 6) {
            glm::vec3 positionOffset, rotationOffset;
            for (int i = 0; i < 3; i++) {
                positionOffset[i] = args::get(poseOffset)[i];
                rotationOffset[i] = args::get(poseOffset)[i + 3];
            }
            camera.setPosition(camera.getPosition() + positionOffset);
            camera.setRotationEuler(camera.getRotationEuler() + rotationOffset);
            camera.updateViewMatrix();
        }

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

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", !showNormals);
        renderer.drawToScreen(toneMapShader);

        if (recording) {
            recorder.captureFrame(camera);
        }

        if (saveImage) {
            static int frameNum = 0;
            std::stringstream ss;
            ss << dataPath << "frame_" << std::setw(6) << std::setfill('0') << frameNum++;
            std::string fileName = ss.str();
            recorder.saveScreenshotToFile(fileName);
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
