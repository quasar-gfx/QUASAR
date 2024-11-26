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
    ForwardRenderer remoteRenderer(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    // "local" scene
    Scene localScene;
    localScene.envCubeMap = remoteScene.envCubeMap;
    PerspectiveCamera localCamera(windowSize.x, windowSize.y);
    localCamera.setViewMatrix(remoteCamera.getViewMatrix());

    // scene with all the meshes
    Scene meshScene;

    QuadsGenerator quadsGenerator(remoteWindowSize);
    MeshFromQuads meshFromQuads(remoteWindowSize);

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
    unsigned int numTriangles = remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS * 2;
    unsigned int maxIndices = numTriangles * 3;
    unsigned int maxVerticesDepth = remoteWindowSize.x * remoteWindowSize.y;

    Mesh mesh = Mesh({
        .numVertices = maxVertices,
        .numIndices = maxIndices,
        .material = new QuadMaterial({ .baseColorTexture = &renderTarget.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    localScene.addChildNode(&node);
    meshScene.addChildNode(&node);

    Node nodeWireframe = Node(&mesh);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    localScene.addChildNode(&nodeWireframe);

    Mesh meshDepth = Mesh({
        .numVertices = maxVerticesDepth,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDepth = Node(&meshDepth);
    nodeDepth.frustumCulled = false;
    nodeDepth.visible = false;
    nodeDepth.primativeType = GL_POINTS;
    localScene.addChildNode(&nodeDepth);

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
    bool saveProxies = false;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool preventCopyingLocalPose = false;
    bool restrictMovementToViewBox = !animationFileIn;
    float viewBoxSize = 0.5f;
    const int intervalValues[] = {0, 25, 50, 100, 200, 500, 1000};
    const char* intervalLabels[] = {"0ms", "25ms", "50ms", "100ms", "200ms", "500ms", "1000ms"};

    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
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

            auto bufferSizes = quadsGenerator.getBufferSizes();

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

            float proxySizeMb = static_cast<float>(totalProxies * sizeof(QuadMapDataPacked)) / BYTES_IN_MB;
            float depthOffsetSizeMb = static_cast<float>(totalDepthOffsets * sizeof(uint16_t)) / BYTES_IN_MB;
            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", totalProxies, proxySizeMb);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", totalDepthOffsets, depthOffsetSizeMb);

            ImGui::Separator();

            glm::vec3 position = localCamera.getPosition();
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                localCamera.setPosition(position);
            }
            glm::vec3 rotation = localCamera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                localCamera.setRotationEuler(rotation);
            }
            ImGui::SliderFloat("Movement Speed", &localCamera.movementSpeed, 0.1f, 20.0f);

            if (ImGui::Checkbox("Show Environment Map", &showEnvMap)) {
                localScene.envCubeMap = showEnvMap ? remoteScene.envCubeMap : nullptr;
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
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::Checkbox("Correct Normal Orientation", &quadsGenerator.doOrientationCorrection)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
            }

            if (ImGui::SliderFloat("Distance Threshold", &quadsGenerator.distanceThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
            }

            if (ImGui::SliderFloat("Angle Threshold", &quadsGenerator.angleThreshold, 0.0f, 180.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
            }

            if (ImGui::SliderFloat("Flat Threshold (x0.01)", &quadsGenerator.flatThreshold, 0.0f, 10.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
            }

            if (ImGui::SliderFloat("Similarity Threshold", &quadsGenerator.proxySimilarityThreshold, 0.0f, 1.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
            }

            ImGui::Separator();

            if (ImGui::SliderFloat("View Box Size", &viewBoxSize, 0.1f, 5.0f)) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

            ImGui::Separator();

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Gen I-Frame", ImVec2(buttonWidth, 0))) {
                generateIFrame = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Gen P-Frame", ImVec2(buttonWidth, 0))) {
                generatePFrame = true;
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
                auto bufferSizes = quadsGenerator.getBufferSizes();

                std::string verticesFileName = dataPath + "vertices.bin";
                std::string indicesFileName = dataPath + "indices.bin";

                // save vertexBuffer
                mesh.vertexBuffer.bind();
                std::vector<Vertex> vertices = mesh.vertexBuffer.getData();
                std::ofstream verticesFile(dataPath + verticesFileName, std::ios::binary);
                verticesFile.write((char*)vertices.data(), bufferSizes.numVertices * sizeof(Vertex));
                verticesFile.close();
                std::cout << "Saved " << bufferSizes.numVertices << " vertices (" <<
                              (float)bufferSizes.numVertices * sizeof(Vertex) / BYTES_IN_MB <<
                              " MB)" << std::endl;

                // save indexBuffer
                mesh.indexBuffer.bind();
                std::vector<unsigned int> indices = mesh.indexBuffer.getData();
                std::ofstream indicesFile(dataPath + indicesFileName, std::ios::binary);
                indicesFile.write((char*)indices.data(), bufferSizes.numIndices * sizeof(unsigned int));
                indicesFile.close();
                std::cout << "Saved " << bufferSizes.numIndices << " indices (" <<
                             (float)bufferSizes.numIndices * sizeof(unsigned int) / BYTES_IN_MB <<
                             " MB)" << std::endl;

                // save color buffer
                renderTarget.saveColorAsPNG(colorFileName);
            }

            if (ImGui::Button("Save Proxies")) {
                preventCopyingLocalPose = true;
                generateIFrame = true;
                saveProxies = true;
            }

            ImGui::End();
        }

        ImGui::Begin("Render Target", 0, flags);
        ImGui::Image((void*)(intptr_t)(renderTarget.colorBuffer.ID), ImVec2(windowSize.x/4, windowSize.y/4), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        localCamera.setAspect(windowSize.x, windowSize.y);
        localCamera.updateProjectionMatrix();
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

                localCamera.processMouseMovement(xoffset, yoffset, true);
            }
        }
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (animator.running) {
            animator.update(dt);
            localCamera.setPosition(animator.getCurrentPosition());
            localCamera.setRotationQuat(animator.getCurrentRotation());
            localCamera.updateViewMatrix();
        }
        else {
            // handle keyboard input
            localCamera.processKeyboard(keys, dt);
        }

        if (rerenderInterval > 0 && now - startRenderTime > rerenderInterval / 1000.0) {
            generateIFrame = true;
            startRenderTime = now;
        }
        if (generateIFrame || generatePFrame) {
            double startTime = glfwGetTime();
            double totalRenderTime = 0.0;
            double totalGenQuadMapTime = 0.0;
            double totalSimplifyTime = 0.0;
            double totalFillQuadsTime = 0.0;
            double totalGetSizeOfProxiesTime = 0.0;
            double totalCreateMeshTime = 0.0;
            double totalGenDepthTime = 0.0;

            if (!preventCopyingLocalPose) {
                remoteCamera.setPosition(localCamera.getPosition());
                remoteCamera.setRotationQuat(localCamera.getRotationQuat());
                remoteCamera.updateViewMatrix();
                std::cout << "Copying local pose to remote pose" << std::endl;
            }

            /*
            ============================
            FIRST PASS: Render the scene to a G-Buffer render target
            ============================
            */
            if (generateIFrame) {
                // render all objects in remoteScene normally
                remoteRenderer.drawObjects(remoteScene, remoteCamera);
            }
            else if (generatePFrame) {
                // render mesh in meshScene into stencil buffer
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();

                // draw old meshes at new remoteCamera view
                remoteRenderer.drawObjects(meshScene, remoteCamera);

                // render mesh in remoteScene using stencil buffer as a mask
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask();

                remoteRenderer.drawObjects(remoteScene, remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                remoteRenderer.pipeline.stencilState.restoreStencilState();
            }
            if (!showNormals) {
                toneMapShader.bind();
                toneMapShader.setBool("toneMap", false); // dont apply tone mapping
                remoteRenderer.drawToRenderTarget(toneMapShader, renderTarget);
            }
            else {
                remoteRenderer.drawToRenderTarget(screenShaderNormals, renderTarget);
            }
            totalRenderTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
            startTime = glfwGetTime();

            /*
            ============================
            SECOND to FOURTH PASSES: Generate quad map and output proxies
            ============================
            */
            quadsGenerator.createProxiesFromGBuffer(remoteRenderer.gBuffer, remoteCamera);

            if (saveProxies) {
                std::string quadsFileName = dataPath + "quads.bin";
                unsigned int savedBytes = quadsGenerator.saveProxies(quadsFileName);
                std::cout << "Saved " << savedBytes << " quads (" << (float)savedBytes / BYTES_IN_MB << " MB)" << std::endl;

                // save color buffer
                std::string colorFileName = dataPath + "color.png";
                renderTarget.saveColorAsPNG(colorFileName);
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
            unsigned int numProxies = quadsGenerator.getBufferSizes().numProxies;

            totalGetSizeOfProxiesTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
            startTime = glfwGetTime();

            meshFromQuads.createMeshFromProxies(
                numProxies, quadsGenerator.depthBufferSize,
                remoteCamera,
                quadsGenerator.outputNormalSphericalsBuffer, quadsGenerator.outputDepthsBuffer,
                quadsGenerator.outputXYsBuffer, quadsGenerator.outputOffsetSizeFlattenedsBuffer,
                quadsGenerator.depthOffsetsBuffer,
                quadsGenerator.getSizesBuffer(),
                mesh
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
            totalGenDepthTime += (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;

            std::cout << "======================================================" << std::endl;
            std::cout << "  Rendering Time: " << totalRenderTime << "ms" << std::endl;
            std::cout << "  Gen Quad Map Time: " << totalGenQuadMapTime << "ms" << std::endl;
            std::cout << "  Simplify Time: " << totalSimplifyTime << "ms" << std::endl;
            std::cout << "  Fill Quads Time: " << totalFillQuadsTime << "ms" << std::endl;
            std::cout << "  Get Size of Proxies Time: " << totalGetSizeOfProxiesTime << "ms" << std::endl;
            std::cout << "  Create Mesh Time: " << totalCreateMeshTime << "ms" << std::endl;
            std::cout << "  Gen Depth Time: " << totalGenDepthTime << "ms" << std::endl;

            preventCopyingLocalPose = false;
            generateIFrame = false;
            generatePFrame = false;
            saveProxies = false;
        }

        nodeWireframe.visible = showWireframe;
        nodeDepth.visible = showDepth;

        if (saveImage && args::get(poseOffset).size() == 6) {
            glm::vec3 positionOffset, rotationOffset;
            for (int i = 0; i < 3; i++) {
                positionOffset[i] = args::get(poseOffset)[i];
                rotationOffset[i] = args::get(poseOffset)[i + 3];
            }
            localCamera.setPosition(localCamera.getPosition() + positionOffset);
            localCamera.setRotationEuler(localCamera.getRotationEuler() + rotationOffset);
            localCamera.updateViewMatrix();
        }

        if (restrictMovementToViewBox) {
            glm::vec3 remotePosition = remoteCamera.getPosition();
            glm::vec3 position = localCamera.getPosition();
            // restrict localCamera position to be inside position +/- viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - viewBoxSize/2, remotePosition.x + viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewBoxSize/2, remotePosition.y + viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewBoxSize/2, remotePosition.z + viewBoxSize/2);
            localCamera.setPosition(position);
            localCamera.updateViewMatrix();
        }

        // render generated meshes
        renderer.pipeline.rasterState.cullFaceEnabled = false;
        renderStats = renderer.drawObjects(localScene, localCamera);
        renderer.pipeline.rasterState.cullFaceEnabled = true;

        // render to screen
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", !showNormals);
        renderer.drawToScreen(toneMapShader);

        if (recording) {
            recorder.captureFrame(localCamera);
        }

        if (saveImage) {
            if (!animationFileIn) {
                glm::vec3 position = localCamera.getPosition();
                glm::vec3 rotation = localCamera.getRotationEuler();
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
                    localCamera.setPosition(glm::vec3(px, py, pz));
                    localCamera.setRotationEuler(glm::vec3(glm::radians(rx), glm::radians(ry), glm::radians(rz)));
                    localCamera.updateViewMatrix();

                    recorder.captureFrame(localCamera);
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
