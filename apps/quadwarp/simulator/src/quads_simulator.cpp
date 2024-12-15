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

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <shaders_common.h>

int main(int argc, char** argv) {
    Config config{};
    config.title = "QuadStream Simulator";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::Flag saveImage(parser, "save", "Take screenshot and exit", {'I', "save-image"});
    args::ValueFlag<std::string> animationFileIn(parser, "path", "Path to camera animation file", {'A', "animation-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
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

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    ForwardRenderer remoteRenderer(config);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteWindowSize.x, remoteWindowSize.y);
    PerspectiveCamera remoteCameraPrev(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    // "local" scene
    std::vector<Scene> localScenes(2);
    localScenes[0].envCubeMap = remoteScene.envCubeMap;
    localScenes[1].envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    // scenes with resulting mesh
    std::vector<Scene> meshScenes(2);
    int currMeshIndex = 0, prevMeshIndex = 1;

    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);

    unsigned int maxVertices = MAX_NUM_PROXIES * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    RenderTarget renderTarget({
        .width = remoteWindowSize.x,
        .height = remoteWindowSize.y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });
    // RenderTarget renderTargetPrev({
    //     .width = remoteWindowSize.x,
    //     .height = remoteWindowSize.y,
    //     .internalFormat = GL_RGBA16F,
    //     .format = GL_RGBA,
    //     .type = GL_HALF_FLOAT,
    //     .wrapS = GL_CLAMP_TO_EDGE,
    //     .wrapT = GL_CLAMP_TO_EDGE,
    //     .minFilter = GL_NEAREST,
    //     .magFilter = GL_NEAREST
    // });

    std::vector<Mesh> meshes; meshes.reserve(2);
    std::vector<Node> nodeMeshes(2);
    std::vector<Node> nodeWireframes(2);
    MeshSizeCreateParams meshParams = {
        .numVertices = maxVertices,
        .numIndices = maxIndices,
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    };
    for (int i = 0; i < 2; i++) {
        meshParams.material = new QuadMaterial({ .baseColorTexture = &renderTarget.colorBuffer });
        meshes.emplace_back(meshParams);
        nodeMeshes[i] = Node(&meshes[i]);
        nodeMeshes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&nodeMeshes[i]);

        nodeWireframes[i] = Node(&meshes[i]);
        nodeWireframes[i].frustumCulled = false;
        nodeWireframes[i].wireframe = true;
        nodeWireframes[i].visible = false;
        nodeWireframes[i].overrideMaterial = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });

        localScenes[i].addChildNode(&nodeMeshes[i]);
        localScenes[i].addChildNode(&nodeWireframes[i]);
    }

    Mesh meshDepth = Mesh({
        .numVertices = maxVerticesDepth,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDepth = Node(&meshDepth);
    nodeDepth.frustumCulled = false;
    nodeDepth.visible = false;
    nodeDepth.primativeType = GL_POINTS;
    localScenes[0].addChildNode(&nodeDepth);
    localScenes[1].addChildNode(&nodeDepth);

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

    bool generateIFrame = true;
    bool generatePFrame = false;
    bool saveToFile = false;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = animationFileIn;
    bool restrictMovementToViewBox = !animationFileIn;
    float viewBoxSize = 0.5f;

    double rerenderInterval = 0.0;
    const int serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool showMeshCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int serverFPSIndex = !animationFileIn ? 0 : 4;

        static bool showEnvMap = true;

        auto quadBufferSizes = quadsGenerator.getBufferSizes();
        auto meshBufferSizes = meshFromQuads.getBufferSizes();

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

            unsigned int totalTriangles = meshBufferSizes.numIndices / 3;
            unsigned int totalProxies = quadBufferSizes.numProxies;
            unsigned int totalDepthOffsets = quadBufferSizes.numDepthOffsets;
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
                localScenes[0].envCubeMap = showEnvMap ? remoteScene.envCubeMap : nullptr;
                localScenes[1].envCubeMap = showEnvMap ? remoteScene.envCubeMap : nullptr;
            }

            if (ImGui::Button("Change Background Color", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                ImGui::OpenPopup("Background Color Popup");
            }
            if (ImGui::BeginPopup("Background Color Popup")) {
                glm::vec4 background = localScenes[0].backgroundColor;
                ImGui::ColorPicker3("Background Color", (float*)&background);
                localScenes[0].backgroundColor = background;
                localScenes[1].backgroundColor = background;
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

            if (ImGui::SliderFloat("View Box Size", &viewBoxSize, 0.1f, 5.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

            ImGui::Separator();

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Gen I-Frame", ImVec2(buttonWidth, 0))) {
                generateIFrame = true;
                runAnimations = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Gen P-Frame", ImVec2(buttonWidth, 0))) {
                generatePFrame = true;
                runAnimations = true;
            }

            ImGui::Combo("Rerender Interval", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels));
            rerenderInterval = 1000.0 / serverFPSValues[serverFPSIndex];

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
                std::vector<Vertex> vertices = meshes[currMeshIndex].vertexBuffer.getData();
                std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                verticesFile.write((char*)vertices.data(), meshBufferSizes.numVertices * sizeof(Vertex));
                verticesFile.close();
                std::cout << "Saved " << meshBufferSizes.numVertices << " vertices (" <<
                              (float)meshBufferSizes.numVertices * sizeof(Vertex) / BYTES_IN_MB <<
                              " MB)" << std::endl;

                // save indexBuffer
                meshes[currMeshIndex].indexBuffer.bind();
                std::vector<unsigned int> indices = meshes[currMeshIndex].indexBuffer.getData();
                std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                indicesFile.write((char*)indices.data(), meshBufferSizes.numIndices * sizeof(unsigned int));
                indicesFile.close();
                std::cout << "Saved " << meshBufferSizes.numIndices << " indices (" <<
                             (float)meshBufferSizes.numIndices * sizeof(unsigned int) / BYTES_IN_MB <<
                             " MB)" << std::endl;

                // save color buffer
                renderTarget.saveColorAsPNG(colorFileName);
            }

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                runAnimations = false;
                saveToFile = true;
            }

            ImGui::End();
        }

        // flags = 0;
        // ImGui::Begin("Prev Color", 0, flags);
        // ImGui::Image((void*)(intptr_t)(renderTargetPrev.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
        // ImGui::End();

        // ImGui::Begin("Current Color", 0, flags);
        // ImGui::Image((void*)(intptr_t)(renderTarget.colorBuffer), ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
        // ImGui::End();
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    double startRenderTime = 0.0;
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

        if (rerenderInterval > 0 && now - startRenderTime > rerenderInterval / 1000.0) {
            generateIFrame = true;
            runAnimations = true;
            startRenderTime = now;
        }
        if (generateIFrame || generatePFrame) {
            double startTime = window->getTime();
            double totalRenderTime = 0.0;
            double totalCreateProxiesTime = 0.0;
            double totalGenQuadMapTime = 0.0;
            double totalSimplifyTime = 0.0;
            double totalFillQuadsTime = 0.0;
            double totalCreateMeshTime = 0.0;
            double totalAppendProxiesMsTime = 0.0f;
            double totalFillOutputQuadsMsTime = 0.0f;
            double totalCreateVertIndTime = 0.0f;
            double totalCreatePFrameTime = 0.0;
            double totalGenDepthTime = 0.0;

            if (!preventCopyingLocalPose) {
                remoteCamera.setPosition(camera.getPosition());
                remoteCamera.setRotationQuat(camera.getRotationQuat());
                remoteCamera.updateViewMatrix();
            }

            auto& remoteCameraToUse = generatePFrame ? remoteCameraPrev : remoteCamera;

            // render all objects in remoteScene normally
            remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);

            // renderTarget.blitToRenderTarget(renderTargetPrev);
            if (!showNormals) {
                toneMapShader.bind();
                toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                remoteRenderer.drawToRenderTarget(toneMapShader, renderTarget);
            }
            else {
                remoteRenderer.drawToRenderTarget(screenShaderNormals, renderTarget);
            }
            totalRenderTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;

            // create proxies from the current frame
            startTime = window->getTime();
            auto sizes = quadsGenerator.createProxiesFromGBuffer(remoteRenderer.gBuffer, remoteCameraToUse);
            unsigned int numProxies = sizes.numProxies;
            totalCreateProxiesTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
            totalGenQuadMapTime += quadsGenerator.stats.timeToGenerateQuadsMs;
            totalSimplifyTime += quadsGenerator.stats.timeToSimplifyQuadsMs;
            totalFillQuadsTime += quadsGenerator.stats.timeToFillOutputQuadsMs;

            // create mesh from proxies
            startTime = glfwGetTime();
            meshFromQuads.appendProxies(numProxies, quadsGenerator.outputQuadBuffers);
            meshFromQuads.createMeshFromProxies(
                numProxies, quadsGenerator.depthOffsets,
                remoteCameraToUse,
                meshes[currMeshIndex]
            );
            totalCreateMeshTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
            totalAppendProxiesMsTime += meshFromQuads.stats.timeToappendProxiesMs;
            totalFillOutputQuadsMsTime += meshFromQuads.stats.timeToFillOutputQuadsMs;
            totalCreateVertIndTime += meshFromQuads.stats.timeToCreateMeshMs;

            if (generatePFrame) {
                startTime = glfwGetTime();
                // at this point, the current mesh is filled with the current frame

                // first, draw the previous mesh at the previous camera view, filling depth buffer
                remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                remoteRenderer.drawObjectsNoLighting(meshScenes[prevMeshIndex], remoteCameraToUse);

                // then, render the current mesh scene into stencil buffer, using the depth buffer from the prev mesh scene
                // this should draw fragments in the current mesh that are not occluded by the prev mesh scene, setting
                // the stencil buffer to 1 where the depth of the curr mesh is the same as the prev mesh scene
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();
                remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
                remoteRenderer.drawObjectsNoLighting(meshScenes[currMeshIndex], remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                // now, render the full remote scene using the stencil buffer as a mask
                // with this, at values where stencil buffer is 1, remoteScene should render
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
                remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                remoteRenderer.drawObjects(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                remoteRenderer.pipeline.stencilState.restoreStencilState();
                totalCreatePFrameTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;

                // create proxies from the new frame
                startTime = glfwGetTime();
                auto sizes = quadsGenerator.createProxiesFromGBuffer(remoteRenderer.gBuffer, remoteCameraToUse);
                unsigned int numProxies = sizes.numProxies;
                totalCreateProxiesTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
                totalGenQuadMapTime += quadsGenerator.stats.timeToGenerateQuadsMs;
                totalSimplifyTime += quadsGenerator.stats.timeToSimplifyQuadsMs;
                totalFillQuadsTime += quadsGenerator.stats.timeToFillOutputQuadsMs;

                // create mesh from proxies
                meshFromQuads.appendProxies(numProxies, quadsGenerator.outputQuadBuffers, false);
                meshFromQuads.createMeshFromProxies(
                    numProxies,
                    quadsGenerator.depthOffsets,
                    remoteCameraToUse,
                    meshes[currMeshIndex]
                );
                totalCreateMeshTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
            }

            // save to file if requested
            if (saveToFile) {
                unsigned int savedBytes;

                startTime = window->getTime();
                savedBytes = quadsGenerator.saveToFile(dataPath + "quads.bin");
                std::cout << "Saved " << savedBytes << " quads (" << (float)savedBytes / BYTES_IN_MB << " MB)" << std::endl;
                std::cout << (window->getTime() - startTime) * MILLISECONDS_IN_SECOND << "ms to save proxies" << std::endl;

                startTime = window->getTime();
                savedBytes = quadsGenerator.saveDepthOffsetsToFile(dataPath + "depthOffsets.bin");
                std::cout << "Saved " << savedBytes << " depth offsets (" << (float)savedBytes / BYTES_IN_MB << " MB)" << std::endl;
                std::cout << (window->getTime() - startTime) * MILLISECONDS_IN_SECOND << "ms to save depth offsets" << std::endl;

                // save color buffer
                std::string colorFileName = dataPath + "color.png";
                renderTarget.saveColorAsPNG(colorFileName);
            }
            currMeshIndex = (currMeshIndex + 1) % 2;
            prevMeshIndex = (prevMeshIndex + 1) % 2;

            // only update the previous camera pose if we are not generating a P-Frame
            if (!generatePFrame) {
                remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
            }

            // For debugging: Generate point cloud from depth map
            if (showDepth) {
                meshFromDepthShader.startTiming();

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
                meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                meshFromDepthShader.endTiming();
                totalGenDepthTime += meshFromDepthShader.getElapsedTime();
            }

            std::cout << "======================================================" << std::endl;
            std::cout << "  Rendering Time: " << totalRenderTime << "ms" << std::endl;
            std::cout << "  Create Proxies Time: " << totalCreateProxiesTime << "ms" << std::endl;
            std::cout << "     Gen Quad Map Time: " << totalGenQuadMapTime << "ms" << std::endl;
            std::cout << "     Simplify Time: " << totalSimplifyTime << "ms" << std::endl;
            std::cout << "     Fill Quads Time: " << totalFillQuadsTime << "ms" << std::endl;
            if (generatePFrame) std::cout << "  P-Frame Creation Time: " << totalCreatePFrameTime << "ms" << std::endl;
            std::cout << "  Create Mesh Time: " << totalCreateMeshTime << "ms" << std::endl;
            std::cout << "     Append Quads Time: " << totalAppendProxiesMsTime << "ms" << std::endl;
            std::cout << "     Fill Output Quads Time: " << totalFillOutputQuadsMsTime << "ms" << std::endl;
            std::cout << "     Create Vert/Ind Time: " << totalCreateVertIndTime << "ms" << std::endl;
            if (showDepth) std::cout << "  Gen Depth Time: " << totalGenDepthTime << "ms" << std::endl;

            preventCopyingLocalPose = false;
            generateIFrame = false;
            generatePFrame = false;
            saveToFile = false;
        }

        for (int i = 0; i < 2; i++) {
            nodeWireframes[i].visible = showWireframe;
        }
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
        renderStats = renderer.drawObjects(localScenes[prevMeshIndex], camera);
        renderer.pipeline.rasterState.cullFaceEnabled = true;

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
