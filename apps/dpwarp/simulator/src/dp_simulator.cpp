#include <iostream>
#include <filesystem>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Utils/Utils.h>
#include <QuadMaterial.h>
#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

int main(int argc, char** argv) {
    Config config{};
    config.title = "Depth Peeling Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size2", "Size of pre-rendered content", {'z', "size2"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> maxLayersIn(parser, "layers", "Max layers", {'n', "max-layers"}, 4);
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'u', "data-path"}, ".");
    args::Flag saveImage(parser, "save", "Take screenshot and exit", {'b', "save-image"});
    args::PositionalList<float> poseOffset(parser, "pose-offset", "Offset for the pose (only used when --save-image is set)");
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
    glm::uvec2 maxProxySize = remoteWindowSize;
    maxProxySize.x = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.x))));
    maxProxySize.y = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.y))));
    int numQuadMaps = glm::log2(static_cast<float>(glm::min(maxProxySize.x, maxProxySize.y))) + 1;

    config.enableVSync = args::get(vsyncIn);
    config.showWindow = !args::get(saveImage);

    std::string scenePath = args::get(scenePathIn);
    std::string dataPath = args::get(dataPathIn) + "/";
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
    DepthPeelingRenderer dpRenderer(config, maxLayers, true);
    ForwardRenderer forwardRenderer(config);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    std::vector<PerspectiveCamera*> remoteCameras(maxViews);
    for (int i = 0; i < maxViews; i++) {
        remoteCameras[i] = new PerspectiveCamera(remoteWindowSize.x, remoteWindowSize.y);
    }
    PerspectiveCamera* centerRemoteCamera = remoteCameras[0];
    SceneLoader loader;
    loader.loadScene(scenePath, remoteScene, *centerRemoteCamera);

    // make last camera have a larger fov
    remoteCameras[maxViews-1]->setFovyDegrees(120.0f);
    remoteCameras[maxViews-1]->setViewMatrix(centerRemoteCamera->getViewMatrix());

    // scene with all the meshes
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(centerRemoteCamera->getViewMatrix());

    struct QuadMapDataPacked {
        glm::uvec2 normalAndFlattenedAndSize; // (normal.xy, normal.z | (flattened | size) << 16)
        float depth;
        glm::vec2 uv;
        unsigned int offset; // offset.xy packed into a single uint
    };
    std::vector<Buffer<glm::uvec2>> normalAndFlattenedAndSizesBuffers(numQuadMaps);
    std::vector<Buffer<float>> depthsBuffers(numQuadMaps);
    std::vector<Buffer<glm::vec2>> uvsBuffers(numQuadMaps);
    std::vector<Buffer<unsigned int>> offsetsBuffers(numQuadMaps);

    std::vector<glm::uvec2> quadMapSizes(numQuadMaps);
    glm::vec2 currQuadMapSize = maxProxySize;
    for (int i = 0; i < numQuadMaps; i++) {
        normalAndFlattenedAndSizesBuffers[i] = Buffer<glm::uvec2>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);
        depthsBuffers[i] = Buffer<float>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);
        uvsBuffers[i] = Buffer<glm::vec2>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);
        offsetsBuffers[i] = Buffer<unsigned int>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);

        quadMapSizes[i] = currQuadMapSize;
        currQuadMapSize /= 2;
    }

    std::vector<Buffer<glm::uvec2>> outputNormalAndFlattenedAndSizesBuffers(maxViews);
    std::vector<Buffer<float>> outputDepthsBuffers(maxViews);
    std::vector<Buffer<glm::vec2>> outputUVsBuffers(maxViews);
    std::vector<Buffer<unsigned int>> outputOffsetsBuffers(maxViews);

    unsigned int maxQuads = remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS;
    for (int view = 0; view < maxViews; view++) {
        outputNormalAndFlattenedAndSizesBuffers[view] = Buffer<glm::uvec2>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
        outputDepthsBuffers[view] = Buffer<float>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
        outputUVsBuffers[view] = Buffer<glm::vec2>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
        outputOffsetsBuffers[view] = Buffer<unsigned int>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
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
    BufferSizes bufferSizes = { 0 };
    std::vector<Buffer<BufferSizes>> sizesBuffers(maxViews);

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

    std::vector<Mesh*> meshDepths(maxViews);
    std::vector<Node*> nodeDepths(maxViews);

    for (int view = 0; view < maxViews; view++) {
        sizesBuffers[view] = Buffer<BufferSizes>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, &bufferSizes);

        meshes[view] = new Mesh({
            .numVertices = maxVertices / (view == 0 || view == maxViews - 1 ? 1 : 4),
            .numIndices = maxIndices / (view == 0 || view == maxViews - 1 ? 1 : 4),
            .material = new QuadMaterial({ .baseColorTexture = &renderTargets[view]->colorBuffer }),
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
    float flatThreshold = 1.0f;
    float proxySimilarityThreshold = 0.25f;
    bool restrictMovementToViewBox = false;
    float viewBoxSize = 0.5f;
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
            unsigned int totalProxies = 0;
            unsigned int totalDepthOffsets = 0;
            for (int view = 0; view < maxViews; view++) {
                if (!nodes[view]->visible) {
                    continue;
                }
                BufferSizes sizes;
                sizesBuffers[view].bind();
                sizesBuffers[view].getSubData(0, 1, &sizes);
                totalTriangles += sizes.numIndices / 3;
                totalProxies += sizes.numProxies;
                totalDepthOffsets += sizes.numDepthOffsets;
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

            float proxySizeMb = static_cast<float>(totalProxies * 8*sizeof(QuadMapDataPacked)) / MB_TO_BITS;
            float depthOffsetSizeMb = static_cast<float>(totalDepthOffsets * 8*sizeof(uint16_t)) / MB_TO_BITS;
            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f Mb)", totalProxies, proxySizeMb);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f Mb)", totalDepthOffsets, depthOffsetSizeMb);

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

            if (ImGui::SliderFloat("Flat Threshold (x0.01)", &flatThreshold, 0.0f, 10.0f)) {
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
                dpRenderer.setViewBoxSize(viewBoxSize);
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
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            std::string fileName = dataPath + std::string(fileNameBase) + "." + time;

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                saveRenderTargetToFile(forwardRenderer, toneMapShader, fileName, windowSize, saveAsHDR);

                for (int view = 1; view < maxViews; view++) {
                    fileName = dataPath + std::string(fileNameBase) + ".view" + std::to_string(view) + "." + time;
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
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            if (ImGui::Button("Save Mesh")) {
                for (int view = 0; view < maxViews; view++) {
                    BufferSizes sizes;
                    sizesBuffers[view].bind();
                    sizesBuffers[view].getSubData(0, 1, &sizes);

                    std::string verticesFileName = dataPath + "vertices" + std::to_string(view) + ".bin";
                    std::string indicesFileName = dataPath + "indices" + std::to_string(view) + ".bin";
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";

                    // save vertexBuffer
                    meshes[view]->vertexBuffer.bind();
                    std::vector<Vertex> vertices = meshes[view]->vertexBuffer.getData();
                    std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), sizes.numVertices * sizeof(Vertex));
                    verticesFile.close();
                    std::cout << "Saved " << sizes.numVertices << " vertices (" <<
                                             sizes.numVertices * 8*sizeof(Vertex) / MB_TO_BITS << " Mb)" << std::endl;

                    // save indexBuffer
                    meshes[view]->indexBuffer.bind();
                    std::vector<unsigned int> indices = meshes[view]->indexBuffer.getData();
                    std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), sizes.numIndices * sizeof(unsigned int));
                    indicesFile.close();
                    std::cout << "Saved " << sizes.numIndices << " indicies (" <<
                                             sizes.numIndices * 8*sizeof(Vertex) / MB_TO_BITS << " Mb)" << std::endl;

                    // save color buffer
                    renderTargets[view]->saveColorAsPNG(colorFileName);
                }
            }

            if (ImGui::Button("Save Proxies")) {
                for (int view = 0; view < maxViews; view++) {
                    BufferSizes sizes;
                    sizesBuffers[view].bind();
                    sizesBuffers[view].getSubData(0, 1, &sizes);

                    std::string quadsFileName = dataPath + "quads" + std::to_string(view) + ".bin";
                    std::ofstream quadsFile(quadsFileName, std::ios::binary);

                    // save number of proxies
                    quadsFile.write((char*)&sizes.numProxies, sizeof(unsigned int));

                    // save proxies
                    outputNormalAndFlattenedAndSizesBuffers[view].bind();
                    std::vector<glm::uvec2> normalAndFlattenedAndSizes(sizes.numProxies);
                    outputNormalAndFlattenedAndSizesBuffers[view].getSubData(0, sizes.numProxies, normalAndFlattenedAndSizes.data());
                    quadsFile.write((char*)normalAndFlattenedAndSizes.data(), sizes.numProxies * sizeof(glm::uvec2));

                    outputDepthsBuffers[view].bind();
                    std::vector<float> depths(sizes.numProxies);
                    outputDepthsBuffers[view].getSubData(0, sizes.numProxies, depths.data());
                    quadsFile.write((char*)depths.data(), sizes.numProxies * sizeof(float));

                    outputUVsBuffers[view].bind();
                    std::vector<glm::vec2> uvs(sizes.numProxies);
                    outputUVsBuffers[view].getSubData(0, sizes.numProxies, uvs.data());
                    quadsFile.write((char*)uvs.data(), sizes.numProxies * sizeof(glm::vec2));

                    outputOffsetsBuffers[view].bind();
                    std::vector<unsigned int> offsets(sizes.numProxies);
                    outputOffsetsBuffers[view].getSubData(0, sizes.numProxies, offsets.data());
                    quadsFile.write((char*)offsets.data(), sizes.numProxies * sizeof(unsigned int));

                    quadsFile.close();
                    std::cout << "Saved " << sizes.numProxies << " quads (" <<
                                sizes.numProxies * 8*sizeof(QuadMapDataPacked) / MB_TO_BITS << " Mb)" << std::endl;

                    // save color buffer
                    std::string colorFileName = dataPath + "color" + std::to_string(view) + ".png";
                    renderTargets[view]->saveColorAsPNG(colorFileName);
                }
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        dpRenderer.resize(width, height);
        forwardRenderer.resize(width, height);

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
            double avgRenderTime = 0.0;
            double avgGenQuadMapTime = 0.0;
            double avgSimplifyTime = 0.0;
            double avgFillQuadsTime = 0.0;
            double avgCreateMeshTime = 0.0;
            double avgGenDepthTime = 0.0;

            /*
            ============================
            FIRST PASS: Render the scene to a G-Buffer render target
            ============================
            */
            dpRenderer.drawObjects(remoteScene, *centerRemoteCamera);
            std::cout << "  Render Time: " << glfwGetTime() - startTime << std::endl;
            startTime = glfwGetTime();

            for (int view = 0; view < maxViews; view++) {
                auto* remoteCamera = remoteCameras[view];

                auto* currMesh = meshes[view];
                auto* currMeshDepth = meshDepths[view];

                /*
                ============================
                FIRST PASS: Render the scene to a G-Buffer render target
                ============================
                */
                if (view < maxViews - 1) {
                    // render to render target
                    if (!showNormals) {
                        toneMapShader.bind();
                        toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                        dpRenderer.peelingLayers[view]->blitToRenderTarget(*renderTargets[view]);
                    }
                    else {
                        dpRenderer.drawToRenderTarget(screenShaderNormals, *renderTargets[view]);
                    }
                }
                // wide fov camera
                else {
                    // render mesh in meshScene into stencil buffer
                    forwardRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();

                    forwardRenderer.drawObjects(meshScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                    // render mesh in remoteScene using stencil buffer as a mask
                    forwardRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask();

                    forwardRenderer.drawObjects(remoteScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    forwardRenderer.pipeline.stencilState.restoreStencilState();

                    // render to render target
                    if (!showNormals) {
                        toneMapShader.bind();
                        toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                        forwardRenderer.drawToRenderTarget(toneMapShader, *renderTargets[view]);
                    }
                    else {
                        forwardRenderer.drawToRenderTarget(screenShaderNormals, *renderTargets[view]);
                    }
                }

                /*
                ============================
                SECOND PASS: Generate quads from G-Buffer
                ============================
                */
                genQuadMapShader.bind();
                {
                    if (view != maxViews - 1) {
                        genQuadMapShader.setTexture(dpRenderer.peelingLayers[view]->normalsBuffer, 0);
                        genQuadMapShader.setTexture(dpRenderer.peelingLayers[view]->depthStencilBuffer, 1);
                    }
                    else {
                        genQuadMapShader.setTexture(forwardRenderer.gBuffer.normalsBuffer, 0);
                        genQuadMapShader.setTexture(forwardRenderer.gBuffer.depthStencilBuffer, 1);
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
                    genQuadMapShader.setFloat("near", remoteCamera->getNear());
                    genQuadMapShader.setFloat("far", remoteCamera->getFar());
                }
                {
                    genQuadMapShader.setBool("doAverageNormal", doAverageNormal);
                    genQuadMapShader.setBool("doOrientationCorrection", doOrientationCorrection);
                    genQuadMapShader.setFloat("distanceThreshold", distanceThreshold);
                    genQuadMapShader.setFloat("angleThreshold", glm::radians(angleThreshold));
                    genQuadMapShader.setFloat("flatThreshold", flatThreshold * 1e-2f);
                }
                {
                    genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffers[view]);

                    genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, normalAndFlattenedAndSizesBuffers[0]);
                    genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, depthsBuffers[0]);
                    genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, uvsBuffers[0]);
                    genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, offsetsBuffers[0]);

                    genQuadMapShader.setImageTexture(0, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetBuffer.internalFormat);
                }
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
                    simplifyQuadMapShader.setFloat("near", remoteCamera->getNear());
                    simplifyQuadMapShader.setFloat("far", remoteCamera->getFar());
                }
                for (int i = 1; i < numQuadMaps; i++) {
                    auto& prevNormalAndFlattenedAndSizeBuffer = normalAndFlattenedAndSizesBuffers[i-1];
                    auto& prevDepthsBuffer = depthsBuffers[i-1];
                    auto& prevUVsBuffer = uvsBuffers[i-1];
                    auto& prevOffsetsBuffer = offsetsBuffers[i-1];

                    auto& currNormalAndFlattenedAndSizeBuffer = normalAndFlattenedAndSizesBuffers[i];
                    auto& currDepthsBuffer = depthsBuffers[i];
                    auto& currUVsBuffer = uvsBuffers[i];
                    auto& currOffsetsBuffer = offsetsBuffers[i];

                    auto& prevQuadMapSize = quadMapSizes[i-1];
                    auto& currQuadMapSize = quadMapSizes[i];

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
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, prevNormalAndFlattenedAndSizeBuffer);
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevDepthsBuffer);
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, prevUVsBuffer);
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, prevOffsetsBuffer);

                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currNormalAndFlattenedAndSizeBuffer);
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currDepthsBuffer);
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currUVsBuffer);
                        simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currOffsetsBuffer);

                        simplifyQuadMapShader.setImageTexture(0, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetBuffer.internalFormat);
                    }
                    simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                   (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                    simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                }

                avgSimplifyTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                /*
                ============================
                FOURTH PASS: Fill output quads buffer
                ============================
                */
                fillOutputQuadsShader.bind();
                for (int i = 0; i < numQuadMaps; i++) {
                    auto& currNormalAndFlattenedAndSizeBuffer = normalAndFlattenedAndSizesBuffers[i];
                    auto& currDepthsBuffer = depthsBuffers[i];
                    auto& currUVsBuffer = uvsBuffers[i];
                    auto& currOffsetsBuffer = offsetsBuffers[i];

                    auto& quadMapSize = quadMapSizes[i];

                    {
                        fillOutputQuadsShader.setVec2("quadMapSize", quadMapSize);
                    }
                    {
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffers[view]);

                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currNormalAndFlattenedAndSizeBuffer);
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currDepthsBuffer);
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currUVsBuffer);
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currOffsetsBuffer);

                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, outputNormalAndFlattenedAndSizesBuffers[view]);
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, outputDepthsBuffers[view]);
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, outputUVsBuffers[view]);
                        fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, outputOffsetsBuffers[view]);
                    }
                    fillOutputQuadsShader.dispatch((quadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                   (quadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                    fillOutputQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                }

                avgFillQuadsTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                /*
                ============================
                FIFTH PASS: Generate mesh from quads
                ============================
                */
                // get output quads size (same as number of proxies)
                BufferSizes sizes;
                sizesBuffers[view].bind();
                sizesBuffers[view].getSubData(0, 1, &sizes);
                unsigned int outputQuadsSize = sizes.numProxies;

                createMeshFromQuadsShader.bind();
                {
                    createMeshFromQuadsShader.setMat4("view", remoteCamera->getViewMatrix());
                    createMeshFromQuadsShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    createMeshFromQuadsShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    createMeshFromQuadsShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));
                    createMeshFromQuadsShader.setFloat("near", remoteCamera->getNear());
                    createMeshFromQuadsShader.setFloat("far", remoteCamera->getFar());
                }
                {
                    createMeshFromQuadsShader.setVec2("remoteWindowSize", remoteWindowSize);
                    createMeshFromQuadsShader.setInt("quadMapSize", outputQuadsSize);
                    createMeshFromQuadsShader.setVec2("depthBufferSize", depthBufferSize);
                }
                {
                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffers[view]);

                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currMesh->vertexBuffer);
                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currMesh->indexBuffer);
                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currMesh->indirectBuffer);

                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, outputNormalAndFlattenedAndSizesBuffers[view]);
                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, outputDepthsBuffers[view]);
                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, outputUVsBuffers[view]);
                    createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, outputOffsetsBuffers[view]);

                    createMeshFromQuadsShader.setImageTexture(0, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetBuffer.internalFormat);
                }
                createMeshFromQuadsShader.dispatch((outputQuadsSize + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
                createMeshFromQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                avgCreateMeshTime += glfwGetTime() - startTime;
                startTime = glfwGetTime();

                /*
                ============================
                For debugging: Generate point cloud from depth map
                ============================
                */
                meshFromDepthShader.bind();
                {
                    if (view != maxViews - 1) {
                        meshFromDepthShader.setTexture(dpRenderer.peelingLayers[view]->depthStencilBuffer, 0);
                    }
                    else {
                        meshFromDepthShader.setTexture(forwardRenderer.gBuffer.depthStencilBuffer, 0);
                    }
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
                meshFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                                  GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                avgGenDepthTime += glfwGetTime() - startTime;
            }

            std::cout << "  Avg Rendering Time: " << avgRenderTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Gen Quad Map Time: " << avgGenQuadMapTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Simplify Time: " << avgSimplifyTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Gen Quads Time: " << avgFillQuadsTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Create Mesh Time: " << avgCreateMeshTime / maxViews << "s" << std::endl;
            std::cout << "  Avg Gen Depth Time: " << avgGenDepthTime / maxViews << "s" << std::endl;

            rerender = false;
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
        forwardRenderer.pipeline.rasterState.cullFaceEnabled = false;
        renderStats = forwardRenderer.drawObjects(scene, camera);
        forwardRenderer.pipeline.rasterState.cullFaceEnabled = true;

        // render to screen
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", !showNormals);
        forwardRenderer.drawToScreen(toneMapShader);

        if (saveImage) {
            glm::vec3 position = camera.getPosition();
            glm::vec3 rotation = camera.getRotationEuler();
            std::string positionStr = to_string_with_precision(position.x) + "_" + to_string_with_precision(position.y) + "_" + to_string_with_precision(position.z);
            std::string rotationStr = to_string_with_precision(rotation.x) + "_" + to_string_with_precision(rotation.y) + "_" + to_string_with_precision(rotation.z);

            std::cout << "Saving output with pose: Position(" << positionStr << ") Rotation(" << rotationStr << ")" << std::endl;

            std::string fileName = dataPath + "screenshot." + positionStr + "_" + rotationStr;
            saveRenderTargetToFile(forwardRenderer, toneMapShader, fileName, windowSize);
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
