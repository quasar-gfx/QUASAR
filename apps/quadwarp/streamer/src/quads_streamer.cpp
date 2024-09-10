#include <iostream>

#include <args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

#define TEXTURE_PREVIEW_SIZE 500

const std::string DATA_PATH = "./";

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size2", "Size of pre-rendered content", {'S', "size2"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
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

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    renderer.pipeline.rasterState.cullFaceEnabled = false;

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene scene;
    Scene remoteScene;
    PerspectiveCamera camera(screenWidth, screenHeight);
    PerspectiveCamera remoteCamera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, remoteScene, remoteCamera);

    camera.setPosition(remoteCamera.getPosition());
    camera.setRotationQuat(remoteCamera.getRotationQuat());
    camera.updateViewMatrix();

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    unsigned int remoteWidth = size2Width;
    unsigned int remoteHeight = size2Height;

    RenderTarget renderTarget({
        .width = remoteWidth,
        .height = remoteHeight,
        .internalFormat = GL_RGBA16,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    int numVertices = remoteWidth * remoteHeight * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    int numVerticesDepth = remoteWidth * remoteHeight;

    int numTriangles = remoteWidth * remoteHeight * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;

    unsigned int zero = 0;
    Buffer<unsigned int> numVerticesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(GLuint), &zero);
    Buffer<unsigned int> numIndicesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(GLuint), &zero);

    struct QuadMapData {
        bool ignore;
        bool flattened;
        glm::vec3 normal;
        glm::vec2 uv;
        float depth;
        glm::ivec2 offset;
        unsigned int size;
    };
    glm::vec2 quadMapBuffer2x2Size = glm::vec2(remoteWidth, remoteHeight);
    Buffer<QuadMapData> quadMapBuffer2x2(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(QuadMapData) * quadMapBuffer2x2Size.x * quadMapBuffer2x2Size.y, nullptr);

    glm::vec2 quadMapBuffer4x4Size = glm::vec2(quadMapBuffer2x2Size.x, quadMapBuffer2x2Size.y) / 2.0f;
    Buffer<QuadMapData> quadMapBuffer4x4(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(QuadMapData) * quadMapBuffer4x4Size.x * quadMapBuffer4x4Size.y, nullptr);

    std::vector<Buffer<QuadMapData>*> quadMaps = {&quadMapBuffer2x2, &quadMapBuffer4x4};
    std::vector<glm::vec2> quadMapSizes = {quadMapBuffer2x2Size, quadMapBuffer4x4Size};

    glm::vec2 depthBufferSize = 4.0f * glm::vec2(quadMapBuffer2x2Size.x, quadMapBuffer2x2Size.y);
    Buffer<float> depthOffsetBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(QuadMapData) * depthBufferSize.x * depthBufferSize.y, nullptr);

    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .diffuseTexture = &renderTarget.colorBuffer }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Mesh meshWireframe = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeWireframe = Node(&meshWireframe);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    scene.addChildNode(&nodeWireframe);

    Mesh meshDepth = Mesh({
        .vertices = std::vector<Vertex>(numVerticesDepth),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .pointcloud = true,
        .pointSize = 7.5f,
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDepth = Node(&meshDepth);
    nodeDepth.frustumCulled = false;
    nodeDepth.visible = false;
    scene.addChildNode(&nodeDepth);

    bool rerender = true;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool doAverageNormal = true;
    bool doOrientationCorrection = true;
    bool preventCopyingLocalPose = false;
    float distanceThreshold = 0.8f;
    float angleThreshold = 45.0f;
    float flattenedThreshold = 0.001f;
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

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);

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

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Draw Calls: %d", renderStats.drawCalls);

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

            if (ImGui::SliderFloat("Flatten Threshold", &flattenedThreshold, 0.0f, 0.1f)) {
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

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(winSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                if (saveAsHDR) {
                    renderer.gBuffer.saveColorAsHDR(fileName + ".hdr");
                }
                else {
                    renderer.gBuffer.saveColorAsPNG(fileName + ".png");
                }
            }

            ImGui::End();
        }

        if (showMeshCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(winSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCaptureWindow);

            std::string verticesFileName = DATA_PATH + "vertices.bin";
            std::string indicesFileName = DATA_PATH + "indices.bin";
            std::string colorFileName = DATA_PATH + "color.png";

            if (ImGui::Button("Save Mesh")) {
                // save vertexBuffer
                std::vector<Vertex> vertices = mesh.vertexBuffer.getData();
                std::ofstream verticesFile(DATA_PATH + verticesFileName, std::ios::binary);
                verticesFile.write((char*)vertices.data(), mesh.vertexBuffer.getSize() * sizeof(Vertex));
                verticesFile.close();

                // save indexBuffer
                std::vector<unsigned int> indices = mesh.indexBuffer.getData();
                std::ofstream indicesFile(DATA_PATH + indicesFileName, std::ios::binary);
                indicesFile.write((char*)indices.data(), mesh.indexBuffer.getSize() * sizeof(unsigned int));
                indicesFile.close();

                // save color buffer
                renderTarget.saveColorAsPNG(colorFileName);
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        renderer.resize(width, height);

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

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
        .computeCodePath = "./shaders/genQuadMap.comp"
    });

    ComputeShader simplifyQuadMapShader({
        .computeCodePath = "./shaders/simplifyQuadMap.comp"
    });

    ComputeShader genQuadsFromQuadMapShader({
        .computeCodePath = "./shaders/genQuadsFromQuadMap.comp"
    });

    ComputeShader genDepthShader({
        .computeCodePath = "./shaders/genDepthPtCloud.comp"
    });

    double startRenderTime = window->getTime();
    app.onRender([&](double now, double dt) {
        // handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;
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
            renderer.drawObjects(remoteScene, remoteCamera);
            if (!showNormals) {
                renderer.drawToRenderTarget(screenShaderColor, renderTarget);
            }
            else {
                renderer.drawToRenderTarget(screenShaderNormals, renderTarget);
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
                genQuadMapShader.setTexture(renderer.gBuffer.normalsBuffer, 0);
                genQuadMapShader.setTexture(renderer.gBuffer.depthStencilBuffer, 1);
            }
            {
                genQuadMapShader.setVec2("remoteWinSize", glm::vec2(remoteWidth, remoteHeight));
                genQuadMapShader.setVec2("quadMapSize", quadMapBuffer2x2Size);
                genQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
            }
            {
                genQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
                genQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                genQuadMapShader.setMat4("viewInverse", glm::inverse(remoteCamera.getViewMatrix()));
                genQuadMapShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
                genQuadMapShader.setFloat("near", remoteCamera.near);
                genQuadMapShader.setFloat("far", remoteCamera.far);
            }
            {
                genQuadMapShader.setBool("doAverageNormal", doAverageNormal);
                genQuadMapShader.setBool("doOrientationCorrection", doOrientationCorrection);
                genQuadMapShader.setFloat("distanceThreshold", distanceThreshold);
                genQuadMapShader.setFloat("angleThreshold", glm::radians(angleThreshold));
                genQuadMapShader.setFloat("flattenedThreshold", flattenedThreshold);
            }
            {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, quadMapBuffer2x2);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, depthOffsetBuffer);
            }

            // set numVertices and numIndices to 0 before running compute shader
            numVerticesBuffer.bind();
            numVerticesBuffer.setSubData(0, 1, &zero);

            numIndicesBuffer.bind();
            numIndicesBuffer.setSubData(0, 1, &zero);

            // run compute shader
            genQuadMapShader.dispatch(remoteWidth / 16, remoteHeight / 16, 1);
            genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            std::cout << "  QuadMap Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            /*
            ============================
            THIRD PASS: Simplify quad map
            ============================
            */
            for (int i = 1; i < quadMaps.size(); i++) {
                auto* prevBufferPtr = quadMaps[i];
                auto* currBufferPtr = quadMaps[i];
                auto currQuadMapSize = quadMapSizes[i];
                auto prevQuadMapSize = quadMapSizes[i];

                simplifyQuadMapShader.bind();
                {
                    simplifyQuadMapShader.setVec2("remoteWinSize", glm::vec2(remoteWidth, remoteHeight));
                    simplifyQuadMapShader.setVec2("inputQuadMapSize", prevQuadMapSize);
                    simplifyQuadMapShader.setVec2("outputQuadMapSize", currQuadMapSize);
                    simplifyQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
                }
                {
                    simplifyQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
                    simplifyQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                    simplifyQuadMapShader.setMat4("viewInverse", glm::inverse(remoteCamera.getViewMatrix()));
                    simplifyQuadMapShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
                    simplifyQuadMapShader.setFloat("near", remoteCamera.near);
                    simplifyQuadMapShader.setFloat("far", remoteCamera.far);
                }
                {
                    simplifyQuadMapShader.setFloat("similarityThreshold", 0.1);
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, *prevBufferPtr);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, *currBufferPtr);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, depthOffsetBuffer);
                }

                // run compute shader
                simplifyQuadMapShader.dispatch(currQuadMapSize.x / 16, currQuadMapSize.y / 16, 1);
                simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }

            std::cout << "  Simplify QuadMap Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            /*
            ============================
            FOURTH PASS: Generate quads from quad map
            ============================
            */
           for (int i = 0; i < quadMaps.size(); i++) {
                auto* bufferPtr = quadMaps[i];
                auto quadMapSize = quadMapSizes[i];

                genQuadsFromQuadMapShader.bind();
                {
                    genQuadsFromQuadMapShader.setVec2("remoteWinSize", glm::vec2(remoteWidth, remoteHeight));
                    genQuadsFromQuadMapShader.setVec2("quadMapSize", quadMapSize);
                    genQuadsFromQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
                }
                {
                    genQuadsFromQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
                    genQuadsFromQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                    genQuadsFromQuadMapShader.setMat4("viewInverse", glm::inverse(remoteCamera.getViewMatrix()));
                    genQuadsFromQuadMapShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
                    genQuadsFromQuadMapShader.setFloat("near", remoteCamera.near);
                    genQuadsFromQuadMapShader.setFloat("far", remoteCamera.far);
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, *bufferPtr);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, numVerticesBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, numIndicesBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mesh.vertexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mesh.indexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, meshWireframe.vertexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, meshWireframe.indexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, depthOffsetBuffer);
                }

                genQuadsFromQuadMapShader.dispatch(quadMapSize.x / 16, quadMapSize.y / 16, 1);
                genQuadsFromQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
           }

            // get number of vertices and indices in mesh
            unsigned int verticesSize;
            numVerticesBuffer.bind();
            numVerticesBuffer.getSubData(0, 1, &verticesSize);
            numVerticesBuffer.setSubData(0, 1, &zero); // reset for next frame

            unsigned int indicesSize;
            numIndicesBuffer.bind();
            numIndicesBuffer.getSubData(0, 1, &indicesSize);
            numIndicesBuffer.setSubData(0, 1, &zero); // reset for next frame

            mesh.resizeBuffers(verticesSize, indicesSize);
            meshWireframe.resizeBuffers(verticesSize, indicesSize);

            std::cout << "  Quads Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            /*
            ============================
            For debugging: Generate point cloud from depth map
            ============================
            */
            genDepthShader.bind();
            {
                genDepthShader.setTexture(renderer.gBuffer.depthStencilBuffer, 0);
            }
            {
                genDepthShader.setVec2("remoteWinSize", glm::vec2(remoteWidth, remoteHeight));
            }
            {
                genDepthShader.setMat4("view", remoteCamera.getViewMatrix());
                genDepthShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                genDepthShader.setMat4("viewInverse", glm::inverse(remoteCamera.getViewMatrix()));
                genDepthShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));

                genDepthShader.setFloat("near", remoteCamera.near);
                genDepthShader.setFloat("far", remoteCamera.far);
            }
            {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meshDepth.vertexBuffer);
            }
            genDepthShader.dispatch(remoteWidth / 16, remoteHeight / 16, 1);
            genDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                         GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

            std::cout << "  Depth Compute Shader Time: " << glfwGetTime() - startTime << "s" << std::endl;

            rerender = false;
        }

        nodeWireframe.visible = showWireframe;
        nodeDepth.visible = showDepth;

        nodeWireframe.setPosition(node.getPosition() - camera.getForwardVector() * 0.001f);
        nodeDepth.setPosition(node.getPosition() - camera.getForwardVector() * 0.0015f);

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        renderer.drawToScreen(screenShaderColor);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
