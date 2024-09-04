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
    config.title = "Multi-Camera Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size2", "Size of pre-rendered content", {'S', "size2"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
    args::ValueFlag<int> maxAdditionalViewsIn(parser, "views", "Max views", {'n', "max-views"}, 8);
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

    int surfelSize = args::get(surfelSizeIn);
    int maxViews = args::get(maxAdditionalViewsIn) + 2; // 0th is standard view, 1st is large fov view

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene remoteScene = Scene();
    std::vector<PerspectiveCamera*> remoteCameras(maxViews);
    for (int i = 0; i < maxViews; i++) {
        remoteCameras[i] = new PerspectiveCamera(screenWidth, screenHeight);
    }
    PerspectiveCamera* centerRemoteCamera = remoteCameras[0];
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, remoteScene, *centerRemoteCamera);

    remoteCameras[1]->setFovy(90.0f);
    remoteCameras[1]->setViewMatrix(centerRemoteCamera->getViewMatrix());

    Scene scene = Scene();
    PerspectiveCamera camera = PerspectiveCamera(screenWidth, screenHeight);
    camera.setViewMatrix(centerRemoteCamera->getViewMatrix());
    camera.updateViewMatrix();

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    unsigned int remoteWidth = size2Width / surfelSize;
    unsigned int remoteHeight = size2Height / surfelSize;

    std::vector<RenderTarget*> renderTargets(maxViews);

    std::vector<GLuint> vertexBuffers(maxViews);
    int numVertices = remoteWidth * remoteHeight * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;

    std::vector<GLuint> vertexBufferDepths(maxViews);
    int numVerticesDepth = remoteWidth * remoteHeight;

    std::vector<GLuint> indexBuffers(maxViews);
    int numTriangles = remoteWidth * remoteHeight * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;

    GLuint zero = 0;
    std::vector<GLuint> numVerticesSSBOs(maxViews);
    std::vector<GLuint> numIndicesSSBOs(maxViews);

    for (int i = 0; i < maxViews; i++) {
        renderTargets[i] = new RenderTarget({
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

        glGenBuffers(1, &numVerticesSSBOs[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, numVerticesSSBOs[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glGenBuffers(1, &numIndicesSSBOs[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, numIndicesSSBOs[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);

    std::vector<Mesh*> meshesWireframe(maxViews);
    std::vector<Node*> nodesWireframe(maxViews);

    std::vector<Mesh*> meshesDepth(maxViews);
    std::vector<Node*> nodesDepth(maxViews);

    for (int i = 0; i < maxViews; i++) {
        meshes[i] = new Mesh({
            .vertices = std::vector<Vertex>(numVertices),
            .indices = std::vector<unsigned int>(indexBufferSize),
            .material = new UnlitMaterial({ .diffuseTexture = &renderTargets[i]->colorBuffer }),
            .wireframe = false
        });
        nodes[i] = new Node(meshes[i]);
        nodes[i]->frustumCulled = false;
        scene.addChildNode(nodes[i]);

        meshesWireframe[i] = new Mesh({
            .vertices = std::vector<Vertex>(numVertices),
            .indices = std::vector<unsigned int>(indexBufferSize),
            .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
            .wireframe = true
        });
        nodesWireframe[i] = new Node(meshesWireframe[i]);
        nodesWireframe[i]->frustumCulled = false;
        scene.addChildNode(nodesWireframe[i]);

        meshesDepth[i] = new Mesh({
            .vertices = std::vector<Vertex>(numVerticesDepth),
            .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
            .wireframe = false,
            .pointcloud = true,
            .pointSize = 7.5f
        });
        nodesDepth[i] = new Node(meshesDepth[i]);
        nodesDepth[i]->frustumCulled = false;
        scene.addChildNode(nodesDepth[i]);
    }

    Scene meshScene = Scene();
    for (int i = 0; i < maxViews; i++) {
        Node* node = new Node(meshes[i]);
        node->frustumCulled = false;
        node->visible = (i == 0);
        meshScene.addChildNode(node);
    }

    bool rerender = true;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool renderWireframe = false;
    bool doAverageNormal = true;
    bool doOrientationCorrection = true;
    bool preventCopyingLocalPose = false;
    bool restrictMovementToViewCell = false;
    float distanceThreshold = 0.8f;
    float angleThreshold = 45.0f;
    float viewCellSize = 3.0f;
    const int intervalValues[] = {0, 100, 200, 500, 1000};
    const char* intervalLabels[] = {"0ms", "100ms", "200ms", "500ms", "1000ms"};
    bool* showViews = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showViews[i] = true;
    }

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showViewPreviews = true;
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
            ImGui::MenuItem("View Previews", 0, &showViewPreviews);
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

            ImGui::Checkbox("Render Wireframe", &renderWireframe);
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

            ImGui::Separator();

            if (ImGui::SliderFloat("View Cell Size", &viewCellSize, 0.1f, 10.0f)) {
                preventCopyingLocalPose = true;
                rerender = true;
            }

            ImGui::Checkbox("Restrict Movement to View Cell", &restrictMovementToViewCell);

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
                ImGui::Checkbox(("Show View " + std::to_string(i)).c_str(), &showViews[i]);
                if ((i + 1) % columns != 0) {
                    ImGui::SameLine();
                }
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

            if (ImGui::Button("Save Mesh")) {
                for (int i = 0; i < maxViews; i++) {
                    std::string verticesFileName = DATA_PATH + "vertices" + std::to_string(i) + ".bin";
                    std::string indicesFileName = DATA_PATH + "indices" + std::to_string(i) + ".bin";
                    std::string colorFileName = DATA_PATH + "color" + std::to_string(i) + ".png";

                    // save vertexBuffer
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, meshes[i]->vertexBuffer);
                    Vertex* vertices = (Vertex*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                    std::ofstream verticesFile(DATA_PATH + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices, numVertices * sizeof(Vertex));
                    verticesFile.close();

                    // save indexBuffer
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, meshes[i]->indexBuffer);
                    GLuint* indices = (GLuint*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                    std::ofstream indicesFile(DATA_PATH + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices, indexBufferSize * sizeof(GLuint));
                    indicesFile.close();

                    // save color buffer
                    renderTargets[i]->saveColorAsPNG(colorFileName);
                }
            }

            ImGui::End();
        }

        if (showViewPreviews) {
            flags = ImGuiWindowFlags_AlwaysAutoResize;

            const int texturePreviewSize = (screenWidth * 2/3) / maxViews;

            for (int i = 0; i < maxViews; i++) {
                int viewIdx = maxViews - i - 1;
                if (showViews[viewIdx]) {
                    ImGui::SetNextWindowPos(ImVec2(screenWidth - (i + 1) * texturePreviewSize - 30, 40), ImGuiCond_FirstUseEver);
                    ImGui::Begin(("View " + std::to_string(viewIdx)).c_str(), 0, flags);
                    ImGui::Image((void*)(intptr_t)(renderTargets[viewIdx]->colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }
            }
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
    Shader screenShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader screenShaderColor({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader screenShaderNormals({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayNormals.frag"
    });

    ComputeShader genQuadsShader({
        .computeCodePath = "./shaders/genQuadsMulti.comp"
    });

    genQuadsShader.bind();
    genQuadsShader.setVec2("screenSize", glm::vec2(remoteWidth, remoteHeight));
    genQuadsShader.setInt("surfelSize", surfelSize);
    genQuadsShader.unbind();

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

        if (restrictMovementToViewCell) {
            glm::vec3 remotePosition = centerRemoteCamera->getPosition();
            glm::vec3 position = camera.getPosition();
            // restrict camera position to be inside position +/- viewCellSize
            position.x = glm::clamp(position.x, remotePosition.x - viewCellSize/2, remotePosition.x + viewCellSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - viewCellSize/2, remotePosition.y + viewCellSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - viewCellSize/2, remotePosition.z + viewCellSize/2);
            camera.setPosition(position);
        }

        if (rerenderInterval > 0 && now - startRenderTime > rerenderInterval / 1000.0) {
            rerender = true;
            startRenderTime = now;
        }
        if (rerender) {
            if (!preventCopyingLocalPose) {
                centerRemoteCamera->setViewMatrix(camera.getViewMatrix());
            }
            preventCopyingLocalPose = false;

            // update other views
            for (int i = 2; i < maxViews; i++) {
                int x = (i & 1) ? -1 : 1;
                int y = (i & 2) ? -1 : 1;
                int z = (i & 4) ? -1 : 1;
                remoteCameras[i]->setPosition(centerRemoteCamera->getPosition() + viewCellSize/2 * glm::vec3(x, y, z));
                remoteCameras[i]->updateViewMatrix();
            }
            remoteCameras[1]->setViewMatrix(centerRemoteCamera->getViewMatrix());
            remoteCameras[1]->setPosition(centerRemoteCamera->getPosition() - viewCellSize/2 * centerRemoteCamera->getForwardVector());

            double startTime = glfwGetTime();

            for (int i = 0; i < maxViews; i++) {
                auto* remoteCamera = remoteCameras[i];

                // center view
                if (i == 0) {
                    // render all objects in remoteScene normally
                    renderer.drawObjects(remoteScene, *remoteCamera);
                }
                // other views
                else {
                    // render mesh in meshScene into stencil buffer
                    renderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();

                    // make all previous meshes visible and everything else invisible
                    for (int j = 1; j < maxViews; j++) {
                        meshScene.children[j]->visible = (j <= i);
                    }
                    renderer.drawObjects(meshScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                    // render mesh in remoteScene using stencil buffer as a mask
                    renderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask();

                    renderer.drawObjects(remoteScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    renderer.pipeline.stencilState.restoreStencilState();
                }

                // render to render target
                if (!showNormals) {
                    renderer.drawToRenderTarget(screenShaderColor, *renderTargets[i]);
                }
                else {
                    renderer.drawToRenderTarget(screenShaderNormals, *renderTargets[i]);
                }

                genQuadsShader.bind();
                {
                    genQuadsShader.setMat4("viewCenter", centerRemoteCamera->getViewMatrix());
                    genQuadsShader.setMat4("projectionCenter", centerRemoteCamera->getProjectionMatrix());
                    genQuadsShader.setMat4("viewInverseCenter", glm::inverse(centerRemoteCamera->getViewMatrix()));
                    genQuadsShader.setMat4("projectionInverseCenter", glm::inverse(centerRemoteCamera->getProjectionMatrix()));
                }
                {
                    genQuadsShader.setMat4("view", remoteCamera->getViewMatrix());
                    genQuadsShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    genQuadsShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    genQuadsShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));

                    genQuadsShader.setFloat("near", remoteCamera->near);
                    genQuadsShader.setFloat("far", remoteCamera->far);
                }
                {
                    genQuadsShader.setBool("doAverageNormal", doAverageNormal);
                    genQuadsShader.setBool("doOrientationCorrection", doOrientationCorrection);
                    genQuadsShader.setFloat("distanceThreshold", distanceThreshold);
                    genQuadsShader.setFloat("angleThreshold", glm::radians(angleThreshold));
                    genQuadsShader.setBool("doDiscardIfOutsideCenterView", i != 0 && i != 1);
                }
                {
                    genQuadsShader.setTexture(renderer.gBuffer.positionBuffer, 0);
                    genQuadsShader.setTexture(renderer.gBuffer.normalsBuffer, 1);
                    genQuadsShader.setTexture(renderer.gBuffer.idBuffer, 2);
                    genQuadsShader.setTexture(renderer.gBuffer.depthStencilBuffer, 3);
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meshes[i]->vertexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshes[i]->indexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, meshesDepth[i]->vertexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, numVerticesSSBOs[i]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, numIndicesSSBOs[i]);
                }

                // set numVertices and numIndices to 0 before running compute shader
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, numVerticesSSBOs[i]);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);

                glBindBuffer(GL_SHADER_STORAGE_BUFFER, numIndicesSSBOs[i]);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);

                // run compute shader
                genQuadsShader.dispatch(remoteWidth / 16, remoteHeight / 16, 1);
                genQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                // get number of vertices and indices in mesh
                unsigned int verticesSize, indicesSize;
                glGetNamedBufferSubData(numVerticesSSBOs[i], 0, sizeof(GLuint), &verticesSize);
                glGetNamedBufferSubData(numIndicesSSBOs[i], 0, sizeof(GLuint), &indicesSize);

                meshes[i]->resizeBuffers(verticesSize, indicesSize);
                meshesDepth[i]->resizeBuffers(verticesSize, verticesSize);
                meshesWireframe[i]->resizeBuffers(verticesSize, indicesSize);
            }

            std::cout << "Total Mesh Creation Time: " << glfwGetTime() - startTime << "s" << std::endl;

            rerender = false;
        }

        for (int i = 0; i < maxViews; i++) {
            bool showView = showViews[i];

            nodes[i]->visible = showView;
            nodesWireframe[i]->visible = showView && renderWireframe;
            nodesDepth[i]->visible = showView && showDepth;

            nodesWireframe[i]->setPosition(nodes[i]->getPosition() - camera.getForwardVector() * 0.001f);
            nodesDepth[i]->setPosition(nodes[i]->getPosition() - camera.getForwardVector() * 0.0015f);
        }

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        renderer.drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
