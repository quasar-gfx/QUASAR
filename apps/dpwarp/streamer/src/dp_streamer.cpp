#include <iostream>

#include <args.hxx>

#include <OpenGLApp.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

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

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene remoteScene;
    std::vector<PerspectiveCamera*> remoteCameras(maxViews);
    for (int i = 0; i < maxViews; i++) {
        remoteCameras[i] = new PerspectiveCamera(screenWidth, screenHeight);
    }
    PerspectiveCamera* centerRemoteCamera = remoteCameras[0];
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, remoteScene, *centerRemoteCamera);

    remoteCameras[maxViews-1]->setFovy(90.0f);
    remoteCameras[maxViews-1]->setViewMatrix(centerRemoteCamera->getViewMatrix());

    Scene scene;
    PerspectiveCamera camera(screenWidth, screenHeight);
    camera.setViewMatrix(centerRemoteCamera->getViewMatrix());

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    unsigned int remoteWidth = size2Width;
    unsigned int remoteHeight = size2Height;

    std::vector<RenderTarget*> renderTargets(maxViews);

    int numVertices = remoteWidth * remoteHeight * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;

    int numVerticesDepth = remoteWidth * remoteHeight;

    int numTriangles = remoteWidth * remoteHeight * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;

    GLuint zero = 0;
    std::vector<Buffer<unsigned int>> numVerticesBuffers(maxViews);
    std::vector<Buffer<unsigned int>> numIndicesBuffers(maxViews);

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

        numVerticesBuffers[i] = Buffer<unsigned int>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(GLuint), &zero);
        numIndicesBuffers[i] = Buffer<unsigned int>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, sizeof(GLuint), &zero);
    }

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);

    std::vector<Mesh*> meshWireframes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

    std::vector<Mesh*> meshDepths(maxViews);
    std::vector<Node*> nodeDepths(maxViews);

    for (int i = 0; i < maxViews; i++) {
        meshes[i] = new Mesh({
            .vertices = std::vector<Vertex>(numVertices),
            .indices = std::vector<unsigned int>(indexBufferSize),
            .material = new UnlitMaterial({ .diffuseTexture = &renderTargets[i]->colorBuffer }),
            .usage = GL_DYNAMIC_DRAW
        });
        nodes[i] = new Node(meshes[i]);
        nodes[i]->frustumCulled = false;
        scene.addChildNode(nodes[i]);

        // primary view color is yellow
        glm::vec4 color = (i == 0) ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) :
                  glm::vec4(fmod(i * 0.6180339887f, 1.0f),
                            fmod(i * 0.9f, 1.0f),
                            fmod(i * 0.5f, 1.0f),
                            1.0f);

        meshWireframes[i] = new Mesh({
            .vertices = std::vector<Vertex>(numVertices),
            .indices = std::vector<unsigned int>(indexBufferSize),
            .material = new UnlitMaterial({ .baseColor = color }),
            .usage = GL_DYNAMIC_DRAW
        });
        nodeWireframes[i] = new Node(meshWireframes[i]);
        nodeWireframes[i]->frustumCulled = false;
        nodeWireframes[i]->wireframe = true;
        scene.addChildNode(nodeWireframes[i]);

        meshDepths[i] = new Mesh({
            .vertices = std::vector<Vertex>(numVerticesDepth),
            .material = new UnlitMaterial({ .baseColor = color }),
            .pointcloud = true,
            .pointSize = 7.5f,
            .usage = GL_DYNAMIC_DRAW
        });
        nodeDepths[i] = new Node(meshDepths[i]);
        nodeDepths[i]->frustumCulled = false;
        scene.addChildNode(nodeDepths[i]);
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

    ComputeShader genQuadsShader({
        .computeCodePath = "./shaders/genQuadsDP.comp"
    });

    ComputeShader genDepthShader({
        .computeCodePath = "./shaders/genDepth.comp"
    });

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
                    dpRenderer.gBuffer.saveColorAsHDR(fileName + ".hdr");
                }
                else {
                    dpRenderer.gBuffer.saveColorAsPNG(fileName + ".png");
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
                    std::vector<Vertex> vertices = meshes[i]->vertexBuffer.getData();
                    std::ofstream verticesFile(DATA_PATH + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices.data(), meshes[i]->vertexBuffer.getSize() * sizeof(Vertex));
                    verticesFile.close();

                    // save indexBuffer
                    std::vector<unsigned int> indices = meshes[i]->indexBuffer.getData();
                    std::ofstream indicesFile(DATA_PATH + indicesFileName, std::ios::binary);
                    indicesFile.write((char*)indices.data(), meshes[i]->indexBuffer.getSize() * sizeof(unsigned int));
                    indicesFile.close();

                    // save color buffer
                    renderTargets[i]->saveColorAsPNG(colorFileName);
                }
            }

            ImGui::End();
        }

        if (showLayerPreviews) {
            flags = ImGuiWindowFlags_AlwaysAutoResize;

            const int texturePreviewSize = (screenWidth * 2/3) / maxViews;

            for (int i = 0; i < maxViews; i++) {
                int layerIdx = maxViews - i - 1;
                if (showLayers[layerIdx]) {
                    ImGui::SetNextWindowPos(ImVec2(screenWidth - (i + 1) * texturePreviewSize - 30, 40), ImGuiCond_FirstUseEver);
                    ImGui::Begin(("View " + std::to_string(layerIdx)).c_str(), 0, flags);
                    ImGui::Image((void*)(intptr_t)(renderTargets[layerIdx]->colorBuffer.ID), ImVec2(texturePreviewSize, texturePreviewSize), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }
            }
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        dpRenderer.resize(width, height);
        forwardRenderer.resize(width, height);

        camera.aspect = (float)screenWidth / (float)screenHeight;
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
                centerRemoteCamera->setViewMatrix(camera.getViewMatrix());
                for (int i = 1; i < maxViews; i++) {
                    remoteCameras[i]->setViewMatrix(centerRemoteCamera->getViewMatrix());
                }
            }
            preventCopyingLocalPose = false;

            double startTime = glfwGetTime();

            // render all objects in remoteScene
            dpRenderer.drawObjects(remoteScene, *centerRemoteCamera);

            for (int i = 0; i < maxViews; i++) {
                auto* remoteCamera = remoteCameras[i];

                // wide fov camera
                if (i == maxViews - 1) {
                    // render mesh in meshScene into stencil buffer
                    forwardRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer();

                    forwardRenderer.drawObjects(meshScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                    // render mesh in remoteScene using stencil buffer as a mask
                    forwardRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask();

                    forwardRenderer.drawObjects(remoteScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    forwardRenderer.pipeline.stencilState.restoreStencilState();

                    // render to render target
                    if (!showNormals) {
                        forwardRenderer.drawToRenderTarget(screenShaderColor, *renderTargets[i]);
                    }
                    else {
                        forwardRenderer.drawToRenderTarget(screenShaderNormals, *renderTargets[i]);
                    }
                }
                else {
                    // render to render target
                    if (!showNormals) {
                        dpRenderer.drawToRenderTarget(screenShaderColor, *renderTargets[i]);
                    }
                    else {
                        dpRenderer.drawToRenderTarget(screenShaderNormals, *renderTargets[i]);
                    }
                    dpRenderer.peelingLayers[i]->blitToRenderTarget(*renderTargets[i]);
                }

                genQuadsShader.bind();
                {
                    genQuadsShader.setVec2("screenSize", glm::vec2(remoteWidth, remoteHeight));
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
                }
                {
                    if (i != maxViews - 1) {
                        genQuadsShader.setTexture(dpRenderer.peelingLayers[i]->positionBuffer, 0);
                        genQuadsShader.setTexture(dpRenderer.peelingLayers[i]->normalsBuffer, 1);
                        genQuadsShader.setTexture(dpRenderer.peelingLayers[i]->idBuffer, 2);
                        genQuadsShader.setTexture(dpRenderer.peelingLayers[i]->depthStencilBuffer, 3);
                    }
                    else {
                        genQuadsShader.setTexture(forwardRenderer.gBuffer.positionBuffer, 0);
                        genQuadsShader.setTexture(forwardRenderer.gBuffer.normalsBuffer, 1);
                        genQuadsShader.setTexture(forwardRenderer.gBuffer.idBuffer, 2);
                        genQuadsShader.setTexture(forwardRenderer.gBuffer.depthStencilBuffer, 3);
                    }
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, numVerticesBuffers[i]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, numIndicesBuffers[i]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, meshes[i]->vertexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, meshes[i]->indexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, meshWireframes[i]->vertexBuffer);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, meshWireframes[i]->indexBuffer);
                }

                // set numVertices and numIndices to 0 before running compute shader
                numVerticesBuffers[i].bind();
                numVerticesBuffers[i].setSubData(0, 1, &zero);

                numIndicesBuffers[i].bind();
                numIndicesBuffers[i].setSubData(0, 1, &zero);

                // run compute shader
                genQuadsShader.dispatch(remoteWidth / 16, remoteHeight / 16, 1);
                genQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                // get number of vertices and indices in mesh
                unsigned int verticesSize;
                numVerticesBuffers[i].bind();
                numVerticesBuffers[i].getSubData(0, 1, &verticesSize);

                unsigned int indicesSize;
                numIndicesBuffers[i].bind();
                numIndicesBuffers[i].getSubData(0, 1, &indicesSize);

                meshes[i]->resizeBuffers(verticesSize, indicesSize);
                meshWireframes[i]->resizeBuffers(verticesSize, indicesSize);

                // create point cloud for depth map
                genDepthShader.bind();
                {
                    genDepthShader.setVec2("screenSize", glm::vec2(remoteWidth, remoteHeight));
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
                    if (i != maxViews - 1) {
                        genDepthShader.setTexture(dpRenderer.peelingLayers[i]->positionBuffer, 0);
                        genDepthShader.setTexture(dpRenderer.peelingLayers[i]->normalsBuffer, 1);
                        genDepthShader.setTexture(dpRenderer.peelingLayers[i]->idBuffer, 2);
                        genDepthShader.setTexture(dpRenderer.peelingLayers[i]->depthStencilBuffer, 3);
                    }
                    else {
                        genDepthShader.setTexture(forwardRenderer.gBuffer.positionBuffer, 0);
                        genDepthShader.setTexture(forwardRenderer.gBuffer.normalsBuffer, 1);
                        genDepthShader.setTexture(forwardRenderer.gBuffer.idBuffer, 2);
                        genDepthShader.setTexture(forwardRenderer.gBuffer.depthStencilBuffer, 3);
                    }
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meshDepths[i]->vertexBuffer);
                }
                genDepthShader.dispatch(remoteWidth / 16, remoteHeight / 16, 1);
                genDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
            }

            std::cout << "Total Mesh Creation Time: " << glfwGetTime() - startTime << "s" << std::endl;

            rerender = false;
        }

        for (int i = 0; i < maxViews; i++) {
            bool showLayer = showLayers[i];

            nodes[i]->visible = showLayer;
            nodeWireframes[i]->visible = showLayer && showWireframe;
            nodeDepths[i]->visible = showLayer && showDepth;

            nodeWireframes[i]->setPosition(nodes[i]->getPosition() - camera.getForwardVector() * 0.001f);
            nodeDepths[i]->setPosition(nodes[i]->getPosition() - camera.getForwardVector() * 0.0015f);
        }

        // render all objects in scene
        forwardRenderer.pipeline.rasterState.cullFaceEnabled = false;
        renderStats = forwardRenderer.drawObjects(scene, camera);
        forwardRenderer.pipeline.rasterState.cullFaceEnabled = true;

        // render to screen
        forwardRenderer.drawToScreen(screenShaderColor);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
