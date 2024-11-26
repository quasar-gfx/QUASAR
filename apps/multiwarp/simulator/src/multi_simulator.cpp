#include <iostream>
#include <filesystem>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Recorder.h>
#include <Animator.h>
#include <Utils/Utils.h>

#include <QuadsGenerator.h>
#include <MeshFromQuads.h>
#include <QuadMaterial.h>
#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

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
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::Flag saveImage(parser, "save", "Take screenshot and exit", {'I', "save-image"});
    args::ValueFlag<std::string> animationFileIn(parser, "path", "Path to camera animation file", {'A', "animation-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
    args::PositionalList<float> poseOffset(parser, "pose-offset", "Offset for the pose (only used when --save-image is set)");
    args::ValueFlag<int> maxAdditionalViewsIn(parser, "views", "Max views", {'n', "max-views"}, 8);
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

    // 0th is standard view, maxViews-1 is large fov view
    int maxAdditionalViews = args::get(maxAdditionalViewsIn);
    int maxViews = !disableWideFov ? maxAdditionalViews + 2 : maxAdditionalViews + 1;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer remoteRenderer(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    std::vector<PerspectiveCamera*> remoteCameras(maxViews);
    for (int view = 0; view < maxViews; view++) {
        remoteCameras[view] = new PerspectiveCamera(remoteWindowSize.x, remoteWindowSize.y);
    }
    PerspectiveCamera* centerRemoteCamera = remoteCameras[0];
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, *centerRemoteCamera);

    // make last camera have a larger fov
    if (!disableWideFov) {
        // make last camera have a larger fov
        remoteCameras[maxViews-1]->setFovyDegrees(120.0f);
    }

    // scene with all the meshes
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(centerRemoteCamera->getViewMatrix());

    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);

    std::vector<RenderTarget> renderTargets; renderTargets.reserve(maxViews);
    RenderTargetCreateParams params = {
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    };
    for (int views = 0; views < maxViews; views++) {
        renderTargets.emplace_back(params);
    }

    unsigned int maxVertices = remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    unsigned int numTriangles = remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS * 2;
    unsigned int maxIndices = numTriangles * 3;

    Scene meshScene;
    std::vector<Mesh*> meshes(maxViews);
    std::vector<Mesh*> meshDepths(maxViews);
    std::vector<Node*> nodeDepths(maxViews);
    std::vector<Node*> nodes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

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

        Node* meshNode = new Node(meshes[view]);
        meshNode->frustumCulled = false;
        meshNode->visible = (view == 0);
        meshScene.addChildNode(meshNode);
    }

    // shaders
    ToneMapShader toneMapShader;

    Shader screenShaderNormals({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYNORMALS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYNORMALS_FRAG_len
    });

    ComputeShader genQuadMapShader({
        .computeCodePath = "shaders/genQuadMap.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader simplifyQuadMapShader({
        .computeCodePath = "shaders/simplifyQuadMap.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader fillOutputQuadsShader({
        .computeCodePath = "shaders/fillOutputQuads.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader createMeshFromQuadsShader({
        .computeCodePath = "shaders/createMeshFromQuads.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
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

    // start recording if headless
    std::ifstream fileStream;
    if (saveImage && animationFileIn) {
        recorder.setOutputPath(dataPath);
        recorder.start();

        fileStream.open(animationFile);
        if (!fileStream.is_open()) {
            std::cerr << "Failed to open file: " << animationFile << std::endl;
            return 1;
        }
    }

    bool rerender = true;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool doOrientationCorrection = true;
    bool preventCopyingLocalPose = false;
    bool saveProxies = false;
    float distanceThreshold = 0.5f;
    float angleThreshold = 85.0f;
    float flatThreshold = 1.0f;
    float proxySimilarityThreshold = 0.25f;
    bool restrictMovementToViewBox = !animationFileIn;
    float viewBoxSize = 0.5f;
    const int intervalValues[] = {0, 25, 50, 100, 200, 500, 1000};
    const char* intervalLabels[] = {"0ms", "25ms", "50ms", "100ms", "200ms", "500ms", "1000ms"};
    bool* showViews = new bool[maxViews];
    for (int view = 0; view < maxViews; ++view) {
        showViews[view] = true;
    }

    unsigned int totalTriangles = 0;
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
        static int intervalIndex = 0;

        static bool showEnvMap = true;

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
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::Checkbox("Correct Normal Orientation", &doOrientationCorrection)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Distance Threshold", &distanceThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Angle Threshold", &angleThreshold, 0.0f, 180.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Flat Threshold (0.01)", &flatThreshold, 0.0f, 10.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Similarity Threshold", &proxySimilarityThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            ImGui::Separator();

            if (ImGui::SliderFloat("View Box Size", &viewBoxSize, 0.1f, 5.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

            ImGui::Separator();

            if (ImGui::Button("Rerender", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                rerender = true;
            }

            if (ImGui::Combo("Rerender Interval", &intervalIndex, intervalLabels, IM_ARRAYSIZE(intervalLabels))) {
                rerenderInterval = intervalValues[intervalIndex];
            }

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
                    verticesFile.write((char*)vertices.data(), maxVertices * sizeof(Vertex));
                    verticesFile.close();
                    std::cout << "Saved " << vertices.size() << " vertices (" <<
                                             (float)vertices.size() * sizeof(Vertex) / BYTES_IN_MB <<
                                             " MB)" << std::endl;

                    // save indexBuffer
                    meshes[view]->indexBuffer.bind();
                    std::vector<unsigned int> indices = meshes[view]->indexBuffer.getData();
                    std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), maxIndices * sizeof(unsigned int));
                    indicesFile.close();
                    std::cout << "Saved " << indices.size() << " indicies (" <<
                                             (float)indices.size() * sizeof(Vertex) / BYTES_IN_MB <<
                                             " MB)" << std::endl;

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    renderTargets[view].saveColorAsPNG(colorFileName);
                }
            }

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                rerender = true;
                saveProxies = true;
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

    double startRenderTime = window->getTime();
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
            animator.update(dt);
            camera.setPosition(animator.getCurrentPosition());
            camera.setRotationQuat(animator.getCurrentRotation());
            camera.updateViewMatrix();
        }
        else {
            // handle keyboard input
            camera.processKeyboard(keys, dt);
        }

        if (rerenderInterval > 0 && now - startRenderTime > rerenderInterval / 1000.0) {
            rerender = true;
            startRenderTime = now;
        }
        if (rerender) {
            totalTriangles = 0;
            totalProxies = 0;
            totalDepthOffsets = 0;

            if (!preventCopyingLocalPose) {
                centerRemoteCamera->setViewMatrix(camera.getViewMatrix());

                // update other views
                for (int view = 1; view < maxViews - 1; view++) {
                    glm::vec3 offset = offsets[view - 1];
                    remoteCameras[view]->setViewMatrix(centerRemoteCamera->getViewMatrix());
                    remoteCameras[view]->setPosition(centerRemoteCamera->getPosition() + viewBoxSize/2 * offset);
                    remoteCameras[view]->updateViewMatrix();
                }
                remoteCameras[maxViews-1]->setViewMatrix(centerRemoteCamera->getViewMatrix());
            }
            preventCopyingLocalPose = false;

            std::cout << "======================================================" << std::endl;

            double startTime = glfwGetTime();
            double totalRenderTime = 0.0;
            double totalGenQuadMapTime = 0.0;
            double totalSimplifyTime = 0.0;
            double totalFillQuadsTime = 0.0;
            double totalGetSizeOfProxiesTime = 0.0;
            double totalCreateMeshTime = 0.0;
            double totalGenDepthTime = 0.0;

            for (int view = 0; view < maxViews; view++) {
                auto* remoteCamera = remoteCameras[view];

                auto* currMesh = meshes[view];
                auto* currMeshDepth = meshDepths[view];
                startTime = glfwGetTime();

                /*
                ============================
                FIRST PASS: Render the scene to a G-Buffer render target
                ============================
                */
                // center view
                if (view == 0) {
                    // render all objects in remoteScene normally
                    remoteRenderer.drawObjects(remoteScene, *remoteCamera);
                }
                // other views
                else {
                    // render mesh in meshScene into stencil buffer
                    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();

                    // make all previous meshes visible and everything else invisible
                    for (int prevView = 1; prevView < maxViews; prevView++) {
                        meshScene.children[prevView]->visible = (prevView < view);
                    }
                    remoteRenderer.drawObjects(meshScene, *remoteCamera);

                    // render mesh in remoteScene using stencil buffer as a mask
                    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask();

                    remoteRenderer.drawObjects(remoteScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    remoteRenderer.pipeline.stencilState.restoreStencilState();
                }

                // render to render target
                if (!showNormals) {
                    toneMapShader.bind();
                    toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                    remoteRenderer.drawToRenderTarget(toneMapShader, renderTargets[view]);
                }
                else {
                    remoteRenderer.drawToRenderTarget(screenShaderNormals, renderTargets[view]);
                }
                totalRenderTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
                startTime = glfwGetTime();

                /*
                ============================
                SECOND to FOURTH PASSES: Generate quad map and output proxies
                ============================
                */
                quadsGenerator.createProxiesFromGBuffer(remoteRenderer.gBuffer, *remoteCamera);

                if (saveProxies) {
                    std::string quadsFileName = dataPath + "quads" + std::to_string(view) + ".bin";
                    unsigned int savedBytes = quadsGenerator.saveProxies(quadsFileName);
                    std::cout << "Saved " << savedBytes << " quads (" << (float)savedBytes / BYTES_IN_MB << " MB)" << std::endl;

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    renderTargets[view].saveColorAsPNG(colorFileName);
                }

                totalGenQuadMapTime += quadsGenerator.stats.timeToGenerateQuadsMs;
                totalSimplifyTime += quadsGenerator.stats.timeToSimplifyQuadsMs;
                totalFillQuadsTime += quadsGenerator.stats.timeToFillOutputQuadsMs;
                startTime = glfwGetTime();

                /*
                ============================
                FIFTH PASS: Generate mesh from quads
                ============================
                */
                // get output quads size (same as number of proxies)
                auto bufferSizes = quadsGenerator.getBufferSizes();
                totalProxies += bufferSizes.numProxies;
                totalDepthOffsets += bufferSizes.numDepthOffsets;

                totalGetSizeOfProxiesTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
                startTime = glfwGetTime();

                meshFromQuads.createMeshFromProxies(
                    bufferSizes.numProxies, quadsGenerator.depthBufferSize,
                    *remoteCamera,
                    quadsGenerator.outputNormalSphericalsBuffer, quadsGenerator.outputDepthsBuffer,
                    quadsGenerator.outputXYsBuffer, quadsGenerator.outputOffsetSizeFlattenedsBuffer,
                    quadsGenerator.depthOffsetsBuffer,
                    quadsGenerator.getSizesBuffer(),
                    *currMesh
                );

                totalCreateMeshTime += meshFromQuads.stats.timeToCreateMeshMs;
                startTime = glfwGetTime();

                /*
                ============================
                For debugging: Generate point cloud from depth map
                ============================
                */
                meshFromDepthShader.bind();
                {
                    meshFromDepthShader.setTexture(remoteRenderer.gBuffer.depthStencilBuffer, 0);
                }
                {
                    meshFromDepthShader.setVec2("depthMapSize", remoteWindowSize);
                }
                {
                    meshFromDepthShader.setMat4("view", remoteCamera->getViewMatrix());
                    meshFromDepthShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    meshFromDepthShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    meshFromDepthShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));

                    meshFromDepthShader.setFloat("near", remoteCamera->getNear());
                    meshFromDepthShader.setFloat("far", remoteCamera->getFar());
                }
                {
                    meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currMeshDepth->vertexBuffer);
                    meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
                }
                meshFromDepthShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                             (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                totalGenDepthTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
            }

            std::cout << "  Rendering Time: " << totalRenderTime << "ms" << std::endl;
            std::cout << "  Gen Quad Map Time: " << totalGenQuadMapTime << "ms" << std::endl;
            std::cout << "  Simplify Time: " << totalSimplifyTime << "ms" << std::endl;
            std::cout << "  Fill Quads Time: " << totalFillQuadsTime << "ms" << std::endl;
            std::cout << "  Get Size of Proxies Time: " << totalGetSizeOfProxiesTime << "ms" << std::endl;
            std::cout << "  Create Mesh Time: " << totalCreateMeshTime << "ms" << std::endl;
            std::cout << "  Gen Depth Time: " << totalGenDepthTime << "ms" << std::endl;

            rerender = false;
            saveProxies = false;
        }

        // hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showView = showViews[view];

            nodes[view]->visible = showView;
            nodeWireframes[view]->visible = showView && showWireframe;
            nodeDepths[view]->visible = showView && showDepth;
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
            glm::vec3 remotePosition = centerRemoteCamera->getPosition();
            glm::vec3 position = camera.getPosition();
            // restrict camera position to be inside position +/- viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - viewBoxSize/2, remotePosition.x + viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewBoxSize/2, remotePosition.y + viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewBoxSize/2, remotePosition.z + viewBoxSize/2);
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        // render all objects in scene
        renderer.pipeline.rasterState.cullFaceEnabled = false;
        renderStats = renderer.drawObjects(scene, camera);
        renderer.pipeline.rasterState.cullFaceEnabled = true;

        // render to screen
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", !showNormals);
        renderer.drawToScreen(toneMapShader);

        if (recording) {
            recorder.captureFrame(camera);
        }

        if (saveImage) {
            if (!animationFileIn) {
                glm::vec3 position = camera.getPosition();
                glm::vec3 rotation = camera.getRotationEuler();
                std::string positionStr = to_string_with_precision(position.x) + "_" + to_string_with_precision(position.y) + "_" + to_string_with_precision(position.z);
                std::string rotationStr = to_string_with_precision(rotation.x) + "_" + to_string_with_precision(rotation.y) + "_" + to_string_with_precision(rotation.z);

                std::cout << "Saving output with pose: Position(" << positionStr << ") Rotation(" << rotationStr << ")" << std::endl;

                std::string fileName = dataPath + "screenshot." + positionStr + "_" + rotationStr;
                recorder.saveScreenshotToFile(fileName);
                window->close();
            }
            else {
                std::string line;
                if (std::getline(fileStream, line)) {
                    std::stringstream ss(line);
                    float px, py, pz;
                    float rx, ry, rz;
                    int64_t timestamp;
                    ss >> px >> py >> pz >> rx >> ry >> rz >> timestamp;
                    camera.setPosition(glm::vec3(px, py, pz));
                    camera.setRotationEuler(glm::vec3(glm::radians(rx), glm::radians(ry), glm::radians(rz)));
                    camera.updateViewMatrix();

                    recorder.captureFrame(camera);
                }
                else {
                    fileStream.close();
                    recorder.stop();
                    window->close();
                }
            }
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
