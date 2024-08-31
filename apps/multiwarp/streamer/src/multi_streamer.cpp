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
    args::ValueFlag<int> maxViewsIn(parser, "views", "Max views", {'n', "max-views"}, 9);
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
    int maxViews = args::get(maxViewsIn);

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
    PerspectiveCamera* centerCamera = remoteCameras[0];
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, remoteScene, *centerCamera);

    Scene scene = Scene();
    PerspectiveCamera camera = PerspectiveCamera(screenWidth, screenHeight);
    camera.setPosition(centerCamera->getPosition());
    camera.setRotationQuat(centerCamera->getRotationQuat());
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

        glGenBuffers(1, &vertexBuffers[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffers[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &vertexBufferDepths[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBufferDepths[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numVerticesDepth * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &indexBuffers[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffers[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    }

    bool rerender = true;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool renderWireframe = false;
    bool doAverageNormal = true;
    bool doOrientationCorrection = true;
    bool preventCopyingLocalPose = false;
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

            if (ImGui::SliderFloat("View Cell Size", &viewCellSize, 0.1f, 5.0f)) {
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
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffers[i]);
                    Vertex* vertices = (Vertex*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                    std::ofstream verticesFile(DATA_PATH + verticesFileName, std::ios::binary);
                    verticesFile.write((char*)vertices, numVertices * sizeof(Vertex));
                    verticesFile.close();

                    // save indexBuffer
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffers[i]);
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

    ComputeShader genMeshShader({
        .computeCodePath = "./shaders/genMesh.comp"
    });

    genMeshShader.bind();
    genMeshShader.setVec2("screenSize", glm::vec2(remoteWidth, remoteHeight));
    genMeshShader.setInt("surfelSize", surfelSize);
    genMeshShader.unbind();

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

    Scene centerMeshScene = Scene();
    Node* centerMeshNode = new Node(meshes[0]);
    centerMeshNode->frustumCulled = false;
    centerMeshScene.addChildNode(centerMeshNode);

    auto enableRenderingIntoStencilBuffer = [&]() {
        renderer.pipeline.stencilState.stencilTestEnabled = true;

        renderer.pipeline.stencilState.stencilFunc = GL_ALWAYS;
        renderer.pipeline.stencilState.stencilRef = 1;
        renderer.pipeline.stencilState.stencilMask = 0xFF;

        renderer.pipeline.stencilState.writeStencilMask = 0xFF;

        renderer.pipeline.stencilState.stencilFail = GL_KEEP;
        renderer.pipeline.stencilState.stencilPassDepthFail = GL_KEEP;
        renderer.pipeline.stencilState.stencilPassDepthPass = GL_REPLACE;
    };

    auto enableRenderingUsingStencilBufferAsMask = [&]() {
        renderer.pipeline.stencilState.stencilFunc = GL_NOTEQUAL;
        renderer.pipeline.stencilState.stencilRef = 1;
        renderer.pipeline.stencilState.stencilMask = 0xFF;

        renderer.pipeline.stencilState.writeStencilMask = 0x00;
    };

    auto disableRenderingIntoStencilBuffer = [&]() {
        renderer.pipeline.stencilState.stencilTestEnabled = false;

        renderer.pipeline.stencilState.stencilFunc = GL_ALWAYS;
        renderer.pipeline.stencilState.stencilRef = 0;
        renderer.pipeline.stencilState.stencilMask = 0xFF;

        renderer.pipeline.stencilState.writeStencilMask = 0xFF;
    };

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
                centerCamera->setPosition(camera.getPosition());
                centerCamera->setRotationQuat(camera.getRotationQuat());
                centerCamera->updateViewMatrix();
            }
            preventCopyingLocalPose = false;

            // update other views
            for (int i = 1; i < maxViews; i++) {
                int x = (i & 1) ? -1 : 1;
                int y = (i & 2) ? -1 : 1;
                int z = (i & 4) ? -1 : 1;
                remoteCameras[i]->setPosition(centerCamera->getPosition() + viewCellSize * glm::vec3(x, y, z));
                remoteCameras[i]->updateViewMatrix();
            }

            double startTime = glfwGetTime();

            for (int i = 0; i < maxViews; i++) {
                auto* remoteCamera = remoteCameras[i];

                // center view
                if (i == 0) {
                    // render all objects in remoteScene into stencil buffer
                    disableRenderingIntoStencilBuffer();
                    renderer.drawObjects(remoteScene, *remoteCamera);
                }
                // other views
                else {
                    // render mesh in centerMeshScene into stencil buffer
                    enableRenderingIntoStencilBuffer();

                    renderer.drawObjects(centerMeshScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

                    // render mesh in remoteScene using stencil buffer as a mask
                    enableRenderingUsingStencilBufferAsMask();

                    renderer.drawObjects(remoteScene, *remoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    disableRenderingIntoStencilBuffer();
                }

                // render to render target
                if (!showNormals) {
                    renderer.drawToRenderTarget(screenShaderColor, *renderTargets[i]);
                }
                else {
                    renderer.drawToRenderTarget(screenShaderNormals, *renderTargets[i]);
                }
                renderer.gBuffer.blitToRenderTarget(*renderTargets[i]);

                genMeshShader.bind();
                {
                    genMeshShader.setMat4("view", remoteCamera->getViewMatrix());
                    genMeshShader.setMat4("projection", remoteCamera->getProjectionMatrix());
                    genMeshShader.setMat4("viewInverse", glm::inverse(remoteCamera->getViewMatrix()));
                    genMeshShader.setMat4("projectionInverse", glm::inverse(remoteCamera->getProjectionMatrix()));

                    genMeshShader.setFloat("near", remoteCamera->near);
                    genMeshShader.setFloat("far", remoteCamera->far);
                }
                {
                    genMeshShader.setBool("doAverageNormal", doAverageNormal);
                    genMeshShader.setBool("doOrientationCorrection", doOrientationCorrection);
                    genMeshShader.setFloat("distanceThreshold", distanceThreshold);
                    genMeshShader.setFloat("angleThreshold", glm::radians(angleThreshold));
                }
                {
                    genMeshShader.setTexture(renderer.gBuffer.positionBuffer, 0);
                    genMeshShader.setTexture(renderer.gBuffer.normalsBuffer, 1);
                    genMeshShader.setTexture(renderer.gBuffer.idBuffer, 2);
                    genMeshShader.setTexture(renderer.gBuffer.depthStencilBuffer, 3);
                }
                {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffers[i]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffers[i]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vertexBufferDepths[i]);
                }
                genMeshShader.dispatch(remoteWidth, remoteHeight, 1);
                genMeshShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                genMeshShader.unbind();

                meshes[i]->setBuffers(vertexBuffers[i], indexBuffers[i]);
                meshesWireframe[i]->setBuffers(vertexBuffers[i], indexBuffers[i]);
                meshesDepth[i]->setBuffers(vertexBufferDepths[i]);
            }

            std::cout << "Rendering Time: " << glfwGetTime() - startTime << "s" << std::endl;

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
