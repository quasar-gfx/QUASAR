#include <iostream>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

const std::string DATA_PATH = "./";

int main(int argc, char** argv) {
    Config config{};
    config.title = "Depth Peeling Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size2", "Size of pre-rendered content", {'S', "size2"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> maxLayersIn(parser, "layers", "Max layers", {'n', "max-layers"}, 8);
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

    // parse size2
    std::string size2Str = args::get(size2In);
    pos = size2Str.find('x');
    int size2Width = std::stoi(size2Str.substr(0, pos));
    int size2Height = std::stoi(size2Str.substr(pos + 1));

    glm::uvec2 remoteWindowSize = glm::uvec2(size2Width, size2Height);

    // make sure maxProxySize is a power of 2
    int maxProxySize = glm::max(remoteWindowSize.x, remoteWindowSize.y);
    maxProxySize = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize))));
    int numQuadMaps = glm::log2(static_cast<float>(maxProxySize));

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    int maxLayers = args::get(maxLayersIn);
    int maxViews = maxLayers + 1;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    DepthPeelingRenderer dpRenderer(config, maxLayers);
    ForwardRenderer forwardRenderer(config);

    glm::uvec2 windowSize = window->getSize();

    Scene remoteScene;
    std::vector<PerspectiveCamera*> remoteCameras(maxViews);
    for (int i = 0; i < maxViews; i++) {
        remoteCameras[i] = new PerspectiveCamera(remoteWindowSize.x, remoteWindowSize.y);
    }
    PerspectiveCamera* centerRemoteCamera = remoteCameras[0];
    SceneLoader loader;
    loader.loadScene(scenePath, remoteScene, *centerRemoteCamera);

    remoteCameras[maxViews-1]->setFovy(90.0f);
    remoteCameras[maxViews-1]->setViewMatrix(centerRemoteCamera->getViewMatrix());

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(centerRemoteCamera->getViewMatrix());

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    struct QuadMapData {
        alignas(16) glm::vec3 normal;
        alignas(16) float depth;
        alignas(16) glm::vec2 uv;
        alignas(16) glm::uvec2 offset;
        alignas(16) uint size;
        alignas(16) bool flattened;
    };
    std::vector<Buffer<QuadMapData>> quadMaps(numQuadMaps);
    std::vector<glm::vec2> quadMapSizes(numQuadMaps);
    glm::vec2 quadMapSize = glm::vec2(maxProxySize);
    for (int i = 0; i < numQuadMaps; i++) {
        quadMaps[i] = Buffer<QuadMapData>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, quadMapSize.x * quadMapSize.y, nullptr);
        quadMapSizes[i] = quadMapSize;
        quadMapSize /= 2.0f;
    }

    glm::uvec2 depthBufferSize = 4u * remoteWindowSize;
    Texture depthOffsetBuffer({
        .width = depthBufferSize.x,
        .height = depthBufferSize.y,
        .internalFormat = GL_R16F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    std::vector<RenderTarget*> renderTargets(maxViews);
    for (int views = 0; views < maxViews; views++) {
        renderTargets[views] = new RenderTarget({
            .width = windowSize.x,
            .height = windowSize.y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        });
    }

    unsigned int maxVertices = remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    unsigned int numTriangles = remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS * 2;
    unsigned int maxIndices = numTriangles * 3;

    struct BufferSizes {
        unsigned int numVertices;
        unsigned int numIndices;
        unsigned int numProxies;
        unsigned int numDepthOffsets;
    };
    unsigned int totalProxies = 0;
    unsigned int totalDepthOffsets = 0;
    BufferSizes bufferSizes = { 0 };
    unsigned int zeros[4] = { 0 };
    Buffer<unsigned int> bufferSizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(BufferSizes), zeros);

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);

    std::vector<Mesh*> meshWireframes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

    std::vector<Mesh*> meshDepths(maxViews);
    std::vector<Node*> nodeDepths(maxViews);

    for (int view = 0; view < maxViews; view++) {
        meshes[view] = new Mesh({
            .numVertices = maxVertices,
            .numIndices = maxIndices,
            .material = new UnlitMaterial({ .diffuseTexture = &renderTargets[view]->colorBuffer }),
            .usage = GL_DYNAMIC_DRAW
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

        meshWireframes[view] = new Mesh({
            .numVertices = maxVertices,
            .numIndices = maxIndices,
            .material = new UnlitMaterial({ .baseColor = color }),
            .usage = GL_DYNAMIC_DRAW
        });
        nodeWireframes[view] = new Node(meshWireframes[view]);
        nodeWireframes[view]->frustumCulled = false;
        nodeWireframes[view]->wireframe = true;
        scene.addChildNode(nodeWireframes[view]);

        meshDepths[view] = new Mesh({
            .numVertices = maxVerticesDepth,
            .pointcloud = true,
            .pointSize = 7.5f,
            .usage = GL_DYNAMIC_DRAW
        });
        nodeDepths[view] = new Node(meshDepths[view]);
        nodeDepths[view]->frustumCulled = false;
        scene.addChildNode(nodeDepths[view]);
    }

    Scene meshScene;
    Node* node = new Node(meshes[0]);
    node->frustumCulled = false;
    meshScene.addChildNode(node);

    // shaders
    Shader screenShaderColor({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader screenShaderNormals({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayNormals.frag"
    });

    ComputeShader genQuadMapShader({
        .computeCodePath = "../../quadwarp/streamer/shaders/genQuadMap.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader simplifyQuadMapShader({
        .computeCodePath = "../../quadwarp/streamer/shaders/simplifyQuadMap.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader genMeshFromQuadMapsShader({
        .computeCodePath = "../../quadwarp/streamer/shaders/genMeshFromQuadMaps.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader genDepthShader({
        .computeCodePath = "../../quadwarp/streamer/shaders/genDepthPtCloud.comp",
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    bool rerender = true;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool doAverageNormal = true;
    bool doOrientationCorrection = true;
    bool preventCopyingLocalPose = false;
    float distanceThreshold = 0.75f;
    float angleThreshold = 45.0f;
    float flatThreshold = 0.5f;
    float proxySimilarityThreshold = 0.1f;
    const int intervalValues[] = {0, 25, 50, 100, 200, 500, 1000};
    const char* intervalLabels[] = {"0ms", "25ms", "50ms", "100ms", "200ms", "500ms", "1000ms"};
    bool* showLayers = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showLayers[i] = true;
    }

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showLayerPreviews = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showMeshCaptureWindow = false;
        static int intervalIndex = 0;

        glm::vec2 winSize = glm::vec2(windowSize.x, windowSize.y);

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
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::MenuItem("Frame Capture", 0, &showCaptureWindow);
            ImGui::MenuItem("Mesh Capture", 0, &showMeshCaptureWindow);
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

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d", totalProxies);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d", totalDepthOffsets);

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

            if (ImGui::Checkbox("Average Normals", &doAverageNormal)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

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

            if (ImGui::SliderFloat("Flat Threshold (x1e-2)", &flatThreshold, 0.0f, 10.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Similarity Threshold", &proxySimilarityThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            ImGui::Separator();

            if (ImGui::Button("Rerender", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                rerender = true;
            }

            if (ImGui::Combo("Rerender Interval", &intervalIndex, intervalLabels, IM_ARRAYSIZE(intervalLabels))) {
                rerenderInterval = intervalValues[intervalIndex];
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
                    ImGui::Image((void*)(intptr_t)(renderTargets[viewIdx]->colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }
            }
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(winSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            std::string fileName = DATA_PATH + std::string(fileNameBase) + "." + time;

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                if (saveAsHDR) {
                    dpRenderer.gBuffer.saveColorAsHDR(fileName + ".hdr");
                }
                else {
                    dpRenderer.gBuffer.saveColorAsPNG(fileName + ".png");
                }

                for (int view = 1; view < maxViews; view++) {
                    fileName = DATA_PATH + std::string(fileNameBase) + ".view" + std::to_string(view) + "." + time;
                    if (saveAsHDR) {
                        renderTargets[view]->saveColorAsHDR(fileName + ".hdr");
                    }
                    else {
                        renderTargets[view]->saveColorAsPNG(fileName + ".png");
                    }
                }
            }

            ImGui::End();
        }

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(winSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            if (ImGui::Button("Save Mesh")) {
                for (int view = 0; view < maxViews; view++) {
                    std::string verticesFileName = DATA_PATH + "vertices" + std::to_string(view) + ".bin";
                    std::string indicesFileName = DATA_PATH + "indices" + std::to_string(view) + ".bin";
                    std::string colorFileName = DATA_PATH + "color" + std::to_string(view) + ".png";

                    // save vertexBuffer
                    meshes[view]->vertexBuffer.bind();
                    std::vector<Vertex> vertices = meshes[view]->vertexBuffer.getData();
                    std::ofstream verticesFile(DATA_PATH + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), meshes[view]->vertexBuffer.getSize() * sizeof(Vertex));
                    verticesFile.close();

                    // save indexBuffer
                    meshes[view]->indexBuffer.bind();
                    std::vector<unsigned int> indices = meshes[view]->indexBuffer.getData();
                    std::ofstream indicesFile(DATA_PATH + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), meshes[view]->indexBuffer.getSize() * sizeof(unsigned int));
                    indicesFile.close();

                    // save color buffer
                    renderTargets[view]->saveColorAsPNG(colorFileName);
                }
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        forwardRenderer.resize(width, height);

        camera.aspect = (float)windowSize.x / (float)windowSize.y;
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

        // handle keyboard input
        auto keys = window->getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (rerenderInterval > 0 && now - startRenderTime > rerenderInterval / 1000.0) {
            rerender = true;
            startRenderTime = now;
        }
        if (rerender) {
            if (!preventCopyingLocalPose) {
                centerRemoteCamera->setViewMatrix(camera.getViewMatrix());
                for (int i = 1; i < maxViews; i++) {
                    remoteCameras[i]->setViewMatrix(centerRemoteCamera->getViewMatrix());
                }
            }
            preventCopyingLocalPose = false;

            std::cout << "======================================================" << std::endl;

            double startTime = glfwGetTime();
            double avgGenQuadMapTime = 0.0;
            double avgSimplifyTime = 0.0;
            double avgGenQuadsTime = 0.0;
            double avgSetMeshBuffersTime = 0.0;
            double avgGenDepthTime = 0.0;
            totalProxies = 0;
            totalDepthOffsets = 0;

            /*
            ============================
            FIRST PASS: Render the scene to a G-Buffer render target
            ============================
            */
            dpRenderer.drawObjects(remoteScene, *centerRemoteCamera);

            std::cout << "  Render Time: " << glfwGetTime() - startTime << std::endl;

            for (int view = 0; view < maxViews; view++) {
                auto* remoteCamera = remoteCameras[view];

                auto* currMesh = meshes[view];
                auto* currMeshDepth = meshDepths[view];
                auto* currMeshWireframe = meshWireframes[view];

                // wide fov camera
                if (view == maxViews - 1) {
                    // render mesh in meshScene into stencil buffer
                    forwardRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();

                    forwardRenderer.drawObjects(meshScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                    // render mesh in remoteScene using stencil buffer as a mask
                    forwardRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask();

                    forwardRenderer.drawObjects(remoteScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    forwardRenderer.pipeline.stencilState.restoreStencilState();

                    // render to render target
                    if (!showNormals) {
                        screenShaderColor.bind();
                        screenShaderColor.setBool("doToneMapping", false); // dont apply tone mapping
                        forwardRenderer.drawToRenderTarget(screenShaderColor, *renderTargets[view]);
                    }
                    else {
                        forwardRenderer.drawToRenderTarget(screenShaderNormals, *renderTargets[view]);
                    }
                }
                else {
                    // render to render target
                    if (!showNormals) {
                        dpRenderer.peelingLayers[view]->blitToRenderTarget(*renderTargets[view]);
                    }
                    else {
                        dpRenderer.drawToRenderTarget(screenShaderNormals, *renderTargets[view]);
                    }
                }
                startTime = glfwGetTime();

                /*
                ============================
                SECOND PASS: Generate quads from G-Buffer
                ============================
                */
                genQuadMapShader.bind();
                {
                    if (view != maxViews - 1) {
                        genDepthShader.setTexture(dpRenderer.peelingLayers[view]->normalsBuffer, 0);
                        genDepthShader.setTexture(dpRenderer.peelingLayers[view]->depthStencilBuffer, 1);
                    }
                    else {
                        genDepthShader.setTexture(forwardRenderer.gBuffer.normalsBuffer, 0);
                        genDepthShader.setTexture(forwardRenderer.gBuffer.depthStencilBuffer, 1);
                    }
                }
                {
                    genQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);
                    genQuadMapShader.setVec2("quadMapSize", quadMapSizes[0]);
                    genQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
                }
                {
                    genQuadMapShader.setMat4("view", remoteCamera->getViewMatrix());
                    genQuadMapShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    genQuadMapShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    genQuadMapShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));
                    genQuadMapShader.setFloat("near", remoteCamera->near);
                    genQuadMapShader.setFloat("far", remoteCamera->far);
                }
                {
                    genQuadMapShader.setBool("doAverageNormal", doAverageNormal);
                    genQuadMapShader.setBool("doOrientationCorrection", doOrientationCorrection);
                    genQuadMapShader.setFloat("distanceThreshold", distanceThreshold);
                    genQuadMapShader.setFloat("angleThreshold", glm::radians(angleThreshold));
                    genQuadMapShader.setFloat("flatThreshold", flatThreshold * 1e-2f);
                    genQuadMapShader.setBool("discardOutOfRangeDepths", true);
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, quadMaps[0]);
                    glBindImageTexture(1, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetBuffer.internalFormat);
                }

                // run compute shader
                genQuadMapShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                          (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

                avgGenQuadMapTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                /*
                ============================
                THIRD PASS: Simplify quad map
                ============================
                */
                simplifyQuadMapShader.bind();
                {
                    simplifyQuadMapShader.setMat4("view", remoteCamera->getViewMatrix());
                    simplifyQuadMapShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    simplifyQuadMapShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    simplifyQuadMapShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));
                    simplifyQuadMapShader.setFloat("near", remoteCamera->near);
                    simplifyQuadMapShader.setFloat("far", remoteCamera->far);
                }
                for (int i = 1; i < quadMaps.size(); i++) {
                    auto& prevBuffer = quadMaps[i-1];
                    auto& currBuffer = quadMaps[i];
                    auto prevQuadMapSize = quadMapSizes[i-1];
                    auto currQuadMapSize = quadMapSizes[i];

                    {
                        simplifyQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);
                        simplifyQuadMapShader.setVec2("inputQuadMapSize", prevQuadMapSize);
                        simplifyQuadMapShader.setVec2("outputQuadMapSize", currQuadMapSize);
                        simplifyQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
                    }
                    {
                        simplifyQuadMapShader.setFloat("flatThreshold", flatThreshold * 1e-2f);
                        simplifyQuadMapShader.setFloat("proxySimilarityThreshold", proxySimilarityThreshold);
                    }
                    {
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, prevBuffer);
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, currBuffer);
                        glBindImageTexture(2, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetBuffer.internalFormat);
                    }

                    simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                   (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                    simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                }

                avgSimplifyTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                /*
                ============================
                FOURTH PASS: Generate meshes from quad map
                ============================
                */
                genMeshFromQuadMapsShader.bind();
                {
                    genMeshFromQuadMapsShader.setMat4("view", remoteCamera->getViewMatrix());
                    genMeshFromQuadMapsShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    genMeshFromQuadMapsShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    genMeshFromQuadMapsShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));
                    genMeshFromQuadMapsShader.setFloat("near", remoteCamera->near);
                    genMeshFromQuadMapsShader.setFloat("far", remoteCamera->far);
                }
                for (int i = 0; i < quadMaps.size(); i++) {
                    auto& quadMap = quadMaps[i];
                    auto quadMapSize = quadMapSizes[i];

                    {
                        genMeshFromQuadMapsShader.setVec2("remoteWindowSize", remoteWindowSize);
                        genMeshFromQuadMapsShader.setVec2("quadMapSize", quadMapSize);
                        genMeshFromQuadMapsShader.setVec2("depthBufferSize", depthBufferSize);
                    }
                    {
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, quadMap);
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferSizesBuffer);
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, currMesh->vertexBuffer);
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, currMesh->indexBuffer);
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, currMeshWireframe->vertexBuffer);
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, currMeshWireframe->indexBuffer);
                        glBindImageTexture(6, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetBuffer.internalFormat);
                    }

                    genMeshFromQuadMapsShader.dispatch((quadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                       (quadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                    genMeshFromQuadMapsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                                             GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
                }

                avgGenQuadsTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                // get number of vertices and indices in mesh
                bufferSizesBuffer.bind();
                bufferSizesBuffer.getSubData(0, 4, &bufferSizes);
                bufferSizesBuffer.setSubData(0, 4, &zeros); // reset for next frame

                currMesh->resizeBuffers(bufferSizes.numVertices, bufferSizes.numIndices);
                currMeshWireframe->resizeBuffers(bufferSizes.numVertices, bufferSizes.numIndices);

                totalProxies += bufferSizes.numProxies;
                totalDepthOffsets += bufferSizes.numDepthOffsets;

                avgSetMeshBuffersTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                /*
                ============================
                For debugging: Generate point cloud from depth map
                ============================
                */
                genDepthShader.bind();
                {
                    if (view != maxViews - 1) {
                        genDepthShader.setTexture(dpRenderer.peelingLayers[view]->depthStencilBuffer, 0);
                    }
                    else {
                        genDepthShader.setTexture(forwardRenderer.gBuffer.depthStencilBuffer, 0);
                    }
                }
                {
                    genDepthShader.setVec2("remoteWindowSize", remoteWindowSize);
                }
                {
                    genDepthShader.setMat4("view", remoteCamera->getViewMatrix());
                    genDepthShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    genDepthShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    genDepthShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));

                    genDepthShader.setFloat("near", remoteCamera->near);
                    genDepthShader.setFloat("far", remoteCamera->far);
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, currMeshDepth->vertexBuffer);
                }
                genDepthShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                        (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                genDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                             GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                avgGenDepthTime += glfwGetTime() - startTime;
            }

            std::cout << "  Avg Gen Quad Map Time: " << avgGenQuadMapTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Simplify Time: " << avgSimplifyTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Gen Quads Time: " << avgGenQuadsTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Set Mesh Buffers Time: " << avgSetMeshBuffersTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Gen Depth Time: " << avgGenDepthTime / maxViews << "s" << std::endl;

            rerender = false;
        }

        // hide/show nodes based on user input
        for (int view = 0; view < maxViews; view++) {
            bool showLayer = showLayers[view];

            nodes[view]->visible = showLayer;
            nodeWireframes[view]->visible = showLayer && showWireframe;
            nodeDepths[view]->visible = showLayer && showDepth;

            nodeWireframes[view]->setPosition(nodes[view]->getPosition() - camera.getForwardVector() * 0.001f);
            nodeDepths[view]->setPosition(nodes[view]->getPosition() - camera.getForwardVector() * 0.0015f);
        }

        // render all objects in scene
        forwardRenderer.pipeline.rasterState.cullFaceEnabled = false;
        renderStats = forwardRenderer.drawObjects(scene, camera);
        forwardRenderer.pipeline.rasterState.cullFaceEnabled = true;

        // render to screen
        screenShaderColor.bind();
        screenShaderColor.setBool("doToneMapping", true);
        forwardRenderer.drawToScreen(screenShaderColor);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
