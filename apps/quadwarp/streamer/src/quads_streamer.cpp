#include <iostream>
#include <filesystem>

#include <args/args.hxx>

#include <OpenGLApp.h>
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
    config.title = "Quads Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size2", "Size of pre-rendered content", {'z', "size2"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
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
    PerspectiveCamera remoteCamera(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(scenePath, remoteScene, remoteCamera);

    // scene with all the meshes
    Scene scene;
    scene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    struct QuadMapDataPacked {
        glm::uvec2 normal;
        float depth;
        glm::vec2 uv;
        unsigned int offset; // offset.xy packed into a single uint
        unsigned int flattenedAndSize; // flattened << 31 | size
    };
    std::vector<Buffer<QuadMapDataPacked>> quadMaps(numQuadMaps);
    std::vector<glm::uvec2> quadMapSizes(numQuadMaps);
    glm::vec2 quadMapSize = maxProxySize;
    for (int i = 0; i < numQuadMaps; i++) {
        quadMaps[i] = Buffer<QuadMapDataPacked>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, quadMapSize.x * quadMapSize.y, nullptr);
        quadMapSizes[i] = quadMapSize;
        quadMapSize /= 2;
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

    RenderTarget renderTarget({
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
    Buffer<BufferSizes> sizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, 1, &bufferSizes);

    Mesh mesh = Mesh({
        .numVertices = maxVertices,
        .numIndices = maxIndices,
        .material = new QuadMaterial({ .baseColorTexture = &renderTarget.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Node nodeWireframe = Node(&mesh);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    scene.addChildNode(&nodeWireframe);

    Mesh meshDepth = Mesh({
        .numVertices = maxVerticesDepth,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDepth = Node(&meshDepth);
    nodeDepth.frustumCulled = false;
    nodeDepth.visible = false;
    nodeDepth.primativeType = GL_POINTS;
    scene.addChildNode(&nodeDepth);

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

    ComputeShader genMeshFromQuadMapsShader({
        .computeCodePath = "shaders/genMeshFromQuadMaps.comp",
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
    float flatThreshold = 2.5f;
    float proxySimilarityThreshold = 0.25f;
    bool restrictMovementToViewBox = false;
    float viewBoxSize = 1.0f;
    const int intervalValues[] = {0, 25, 50, 100, 200, 500, 1000};
    const char* intervalLabels[] = {"0ms", "25ms", "50ms", "100ms", "200ms", "500ms", "1000ms"};

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showMeshCaptureWindow = false;
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

            sizesBuffer.bind();
            sizesBuffer.getSubData(0, 1, &bufferSizes);

            unsigned int totalTriangles = bufferSizes.numIndices / 3;
            unsigned int totalProxies = bufferSizes.numProxies;
            unsigned int totalDepthOffsets = bufferSizes.numDepthOffsets;
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

            if (ImGui::SliderFloat("Flat Threshold (x0.1)", &flatThreshold, 0.0f, 10.0f)) {
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

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = dataPath + std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                saveRenderTargetToFile(renderer, toneMapShader, fileName, windowSize, saveAsHDR);
            }

            ImGui::End();
        }

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            std::string verticesFileName = dataPath + "vertices.bin";
            std::string indicesFileName = dataPath + "indices.bin";
            std::string colorFileName = dataPath + "color.png";

            if (ImGui::Button("Save Mesh")) {
                sizesBuffer.bind();
                sizesBuffer.getSubData(0, 1, &bufferSizes);

                // save vertexBuffer
                mesh.vertexBuffer.bind();
                std::vector<Vertex> vertices = mesh.vertexBuffer.getData();
                std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                verticesFile.write((char*)vertices.data(), bufferSizes.numVertices * sizeof(Vertex));
                verticesFile.close();

                // save indexBuffer
                mesh.indexBuffer.bind();
                std::vector<unsigned int> indices = mesh.indexBuffer.getData();
                std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                indicesFile.write((char*)indices.data(), bufferSizes.numIndices * sizeof(unsigned int));
                indicesFile.close();

                // save color buffer
                renderTarget.saveColorAsPNG(colorFileName);
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.resize(windowSize.x, windowSize.y);

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
                remoteCamera.setPosition(camera.getPosition());
                remoteCamera.setRotationQuat(camera.getRotationQuat());
                remoteCamera.updateViewMatrix();
            }
            preventCopyingLocalPose = false;

            std::cout << "======================================================" << std::endl;

            double startTime = glfwGetTime();

            /*
            ============================
            FIRST PASS: Render the scene to a G-Buffer render target
            ============================
            */
            remoteRenderer.drawObjects(remoteScene, remoteCamera);
            if (!showNormals) {
                toneMapShader.bind();
                toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                remoteRenderer.drawToRenderTarget(toneMapShader, renderTarget);
            }
            else {
                remoteRenderer.drawToRenderTarget(screenShaderNormals, renderTarget);
            }

            std::cout << "  Rendering Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            /*
            ============================
            SECOND PASS: Generate quads from G-Buffer
            ============================
            */
            genQuadMapShader.bind();
            {
                genQuadMapShader.setTexture(remoteRenderer.gBuffer.normalsBuffer, 0);
                genQuadMapShader.setTexture(remoteRenderer.gBuffer.depthStencilBuffer, 1);
            }
            {
                genQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);
                genQuadMapShader.setVec2("quadMapSize", quadMapSizes[0]);
                genQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
            }
            {
                genQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
                genQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                genQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
                genQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
                genQuadMapShader.setFloat("near", remoteCamera.getNear());
                genQuadMapShader.setFloat("far", remoteCamera.getFar());
            }
            {
                genQuadMapShader.setBool("doAverageNormal", doAverageNormal);
                genQuadMapShader.setBool("doOrientationCorrection", doOrientationCorrection);
                genQuadMapShader.setFloat("distanceThreshold", distanceThreshold);
                genQuadMapShader.setFloat("angleThreshold", glm::radians(angleThreshold));
                genQuadMapShader.setFloat("flatThreshold", flatThreshold * 0.1f);
            }
            {
                genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, quadMaps[0]);
                genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, sizesBuffer);
                genQuadMapShader.setImageTexture(0, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetBuffer.internalFormat);
            }

            // run compute shader
            genQuadMapShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                      (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
            genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            std::cout << "  QuadMap Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            /*
            ============================
            THIRD PASS: Simplify quad map
            ============================
            */
            simplifyQuadMapShader.bind();
            {
                simplifyQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
                simplifyQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                simplifyQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
                simplifyQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
                simplifyQuadMapShader.setFloat("near", remoteCamera.getNear());
                simplifyQuadMapShader.setFloat("far", remoteCamera.getFar());
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
                    simplifyQuadMapShader.setFloat("flatThreshold", flatThreshold * 0.1f);
                    simplifyQuadMapShader.setFloat("proxySimilarityThreshold", proxySimilarityThreshold);
                }
                {
                    simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, prevBuffer);
                    simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currBuffer);
                    simplifyQuadMapShader.setImageTexture(0, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetBuffer.internalFormat);
                }

                simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                               (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            }

            std::cout << "  Simplify QuadMap Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            /*
            ============================
            FOURTH PASS: Generate meshes from quad map
            ============================
            */
            genMeshFromQuadMapsShader.bind();
            {
                genMeshFromQuadMapsShader.setMat4("view", remoteCamera.getViewMatrix());
                genMeshFromQuadMapsShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                genMeshFromQuadMapsShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
                genMeshFromQuadMapsShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
                genMeshFromQuadMapsShader.setFloat("near", remoteCamera.getNear());
                genMeshFromQuadMapsShader.setFloat("far", remoteCamera.getFar());
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
                    genMeshFromQuadMapsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, quadMap);
                    genMeshFromQuadMapsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, sizesBuffer);
                    genMeshFromQuadMapsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, mesh.vertexBuffer);
                    genMeshFromQuadMapsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, mesh.indexBuffer);
                    genMeshFromQuadMapsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, mesh.indirectBuffer);
                    genMeshFromQuadMapsShader.setImageTexture(0, depthOffsetBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetBuffer.internalFormat);
                }

                genMeshFromQuadMapsShader.dispatch((quadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                                   (quadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                genMeshFromQuadMapsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
            }

            std::cout << "  Quads Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;
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
            meshFromDepthShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                         (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
            meshFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                              GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

            std::cout << "  Depth Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;

            rerender = false;
        }

        nodeWireframe.visible = showWireframe;
        nodeDepth.visible = showDepth;

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
            glm::vec3 remotePosition = remoteCamera.getPosition();
            glm::vec3 position = camera.getPosition();
            // restrict camera position to be inside position +/- viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - viewBoxSize/2, remotePosition.x + viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewBoxSize/2, remotePosition.y + viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewBoxSize/2, remotePosition.z + viewBoxSize/2);
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        // render generated meshes
        renderer.pipeline.rasterState.cullFaceEnabled = false;
        renderStats = renderer.drawObjects(scene, camera);
        renderer.pipeline.rasterState.cullFaceEnabled = true;

        // render to screen
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", !showNormals);
        renderer.drawToScreen(toneMapShader);

        if (saveImage) {
            glm::vec3 position = camera.getPosition();
            glm::vec3 rotation = camera.getRotationEuler();
            std::string positionStr = to_string_with_precision(position.x) + "_" + to_string_with_precision(position.y) + "_" + to_string_with_precision(position.z);
            std::string rotationStr = to_string_with_precision(rotation.x) + "_" + to_string_with_precision(rotation.y) + "_" + to_string_with_precision(rotation.z);

            std::cout << "Saving output with pose: Position(" << positionStr << ") Rotation(" << rotationStr << ")" << std::endl;

            std::string fileName = dataPath + "screenshot." + positionStr + "_" + rotationStr;
            saveRenderTargetToFile(renderer, toneMapShader, fileName, windowSize);
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
