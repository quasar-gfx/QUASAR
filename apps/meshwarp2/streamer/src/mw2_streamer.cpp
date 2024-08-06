#include <iostream>

#include <args.hxx>

#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

#define TEXTURE_PREVIEW_SIZE 500

const std::string DATA_PATH = "./";

int surfelSize = 4;

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp2 Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size2", "Size of pre-rendered content", {'S', "size2"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
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

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene scene = Scene();
    Scene remoteScene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);
    Camera remoteCamera = Camera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, remoteScene, remoteCamera);

    camera.setPosition(remoteCamera.getPosition());
    camera.setRotationQuat(remoteCamera.getRotationQuat());
    camera.updateViewMatrix();

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    unsigned int remoteWidth = size2Width / surfelSize;
    unsigned int remoteHeight = size2Height / surfelSize;

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

    GLuint vertexBuffer;
    int numVertices = remoteWidth * remoteHeight * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);

    GLuint vertexBufferDepth;
    int numVerticesDepth = remoteWidth * remoteHeight;
    glGenBuffers(1, &vertexBufferDepth);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBufferDepth);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVerticesDepth * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);

    GLuint indexBuffer;
    int numTriangles = remoteWidth * remoteHeight * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);

    bool rerender = true;
    int rerenderInterval = 0;
    bool showDepth = false;
    bool showNormals = false;
    bool renderWireframe = false;
    bool doAverageNormal = true;
    bool doOrientationCorrection = true;
    bool dontCopyCameraPose = false;
    float distanceThreshold = 0.8f;
    float angleThreshold = 45.0f;
    RenderStats renderStats;
    const int intervalValues[] = {0, 100, 200, 500, 1000};
    const char* intervalLabels[] = {"0ms", "100ms", "200ms", "500ms", "1000ms"};
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
                dontCopyCameraPose = true;
                rerender = true;
            }

            ImGui::Separator();

            ImGui::Checkbox("Render Wireframe", &renderWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::Checkbox("Average Normals", &doAverageNormal)) {
                dontCopyCameraPose = true;
                rerender = true;
            }

            if (ImGui::Checkbox("Correct Normal Orientation", &doOrientationCorrection)) {
                dontCopyCameraPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Distance Threshold", &distanceThreshold, 0.0f, 1.0f)) {
                dontCopyCameraPose = true;
                rerender = true;
            }

            if (ImGui::SliderFloat("Angle Threshold", &angleThreshold, 0.0f, 180.0f)) {
                dontCopyCameraPose = true;
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
                    app.renderer->gBuffer.saveColorAsHDR(fileName + ".hdr");
                }
                else {
                    app.renderer->gBuffer.saveColorAsPNG(fileName + ".png");
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
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
                Vertex* vertices = (Vertex*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                std::ofstream verticesFile(DATA_PATH + verticesFileName, std::ios::binary);
                verticesFile.write((char*)vertices, numVertices * sizeof(Vertex));
                verticesFile.close();

                // save indexBuffer
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
                GLuint* indices = (GLuint*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                std::ofstream indicesFile(DATA_PATH + indicesFileName, std::ios::binary);
                indicesFile.write((char*)indices, indexBufferSize * sizeof(GLuint));
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

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    // shaders
    Shader screenShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader screenShader2Color = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader screenShader2Normals = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayNormals.frag"
    });

    ComputeShader genMeshShader({
        .computeCodePath = "./shaders/genMesh2.comp"
    });

    genMeshShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vertexBufferDepth);
    genMeshShader.setVec2("screenSize", glm::vec2(remoteWidth, remoteHeight));
    genMeshShader.setInt("surfelSize", surfelSize);
    genMeshShader.unbind();

    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .diffuseTextureID = renderTarget.colorBuffer.ID }),
        .wireframe = false
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Mesh meshWireframe = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .wireframe = true
    });
    Node nodeWireframe = Node(&meshWireframe);
    nodeWireframe.frustumCulled = false;
    scene.addChildNode(&nodeWireframe);

    Mesh meshDepth = Mesh({
        .vertices = std::vector<Vertex>(numVerticesDepth),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .wireframe = false,
        .pointcloud = true,
        .pointSize = 7.5f
    });
    Node nodeDepth = Node(&meshDepth);
    nodeDepth.frustumCulled = false;
    scene.addChildNode(&nodeDepth);

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
            if (!dontCopyCameraPose) {
                remoteCamera.setPosition(camera.getPosition());
                remoteCamera.setRotationQuat(camera.getRotationQuat());
                remoteCamera.updateViewMatrix();
            }
            dontCopyCameraPose = false;

            double startTime = glfwGetTime();

            // render all objects in remoteScene
            app.renderer->drawObjects(remoteScene, remoteCamera);

            // render to render target
            if (!showNormals) {
                app.renderer->drawToRenderTarget(screenShader2Color, renderTarget);
            }
            else {
                app.renderer->drawToRenderTarget(screenShader2Normals, renderTarget);
            }

            genMeshShader.bind();
            genMeshShader.setMat4("view", remoteCamera.getViewMatrix());
            genMeshShader.setMat4("projection", remoteCamera.getProjectionMatrix());
            genMeshShader.setMat4("viewInverse", glm::inverse(remoteCamera.getViewMatrix()));
            genMeshShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
            genMeshShader.setFloat("near", remoteCamera.near);
            genMeshShader.setFloat("far", remoteCamera.far);
            genMeshShader.setBool("doAverageNormal", doAverageNormal);
            genMeshShader.setBool("doOrientationCorrection", doOrientationCorrection);
            genMeshShader.setFloat("distanceThreshold", distanceThreshold);
            genMeshShader.setFloat("angleThreshold", glm::radians(angleThreshold));
            app.renderer->gBuffer.positionBuffer.bind(0);
            app.renderer->gBuffer.normalsBuffer.bind(1);
            app.renderer->gBuffer.idBuffer.bind(2);
            app.renderer->gBuffer.depthBuffer.bind(3);
            genMeshShader.dispatch(remoteWidth, remoteHeight, 1);
            genMeshShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            genMeshShader.unbind();

            std::cout << "Rendering Time: " << glfwGetTime() - startTime << "s" << std::endl;

            mesh.setBuffers(vertexBuffer, indexBuffer);
            meshWireframe.setBuffers(vertexBuffer, indexBuffer);
            meshDepth.setBuffers(vertexBufferDepth);

            rerender = false;
        }

        nodeWireframe.visible = renderWireframe;
        nodeDepth.visible = showDepth;

        nodeWireframe.setPosition(node.getPosition() - camera.getForwardVector() * 0.001f);
        nodeDepth.setPosition(node.getPosition() - camera.getForwardVector() * 0.0015f);

        // render all objects in scene
        renderStats = app.renderer->drawObjects(scene, camera);

        // render to screen
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
