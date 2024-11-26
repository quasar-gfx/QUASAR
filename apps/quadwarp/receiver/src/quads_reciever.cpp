#include <iostream>

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

#include <MeshFromQuads.h>
#include <QuadMaterial.h>
#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

const std::string DATA_PATH = "../simulator/";

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::Flag loadProxies(parser, "load-proxies", "Load proxies from quads.bin", {'m', "load-proxies"});
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

    config.enableVSync = args::get(vsyncIn);

    std::string sceneFile = args::get(sceneFileIn);
    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    remoteCamera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    remoteCamera.updateViewMatrix();

    // shaders
    ToneMapShader toneMapShader;

    Recorder recorder(renderer, toneMapShader, config.targetFramerate);

    MeshFromQuads meshFromQuads(windowSize);

    std::string colorFileName = DATA_PATH + "color.png";
    Texture colorTexture = Texture({
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .flipVertically = true,
        .path = colorFileName
    });

    Mesh* mesh;

    unsigned int totalTriangles = -1;
    unsigned int totalProxies = -1;
    unsigned int totalDepthOffsets = -1;

    unsigned int maxQuads = windowSize.x * windowSize.y * NUM_SUB_QUADS;
    Buffer<unsigned int> inputNormalSphericalsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
    Buffer<float> inputDepthsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
    Buffer<unsigned int> inputUVsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);
    Buffer<unsigned int> inputOffsetSizeFlattenedsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr);

    double startTime = glfwGetTime();
    double loadFromFilesTime = 0.0;
    double createMeshTime = 0.0;
    if (!args::get(loadProxies)) {
        std::string verticesFileName = DATA_PATH + "vertices.bin";
        std::string indicesFileName = DATA_PATH + "indices.bin";

        auto vertexData = FileIO::loadBinaryFile(verticesFileName);
        auto indexData = FileIO::loadBinaryFile(indicesFileName);

        loadFromFilesTime = glfwGetTime() - startTime;
        startTime = glfwGetTime();

        std::vector<Vertex> vertices(vertexData.size() / sizeof(Vertex));
        std::memcpy(vertices.data(), vertexData.data(), vertexData.size());

        std::vector<unsigned int> indices(indexData.size() / sizeof(unsigned int));
        std::memcpy(indices.data(), indexData.data(), indexData.size());

        mesh = new Mesh({
            .vertices = vertices,
            .indices = indices,
            .material = new QuadMaterial({ .baseColorTexture = &colorTexture }),
        });

        totalTriangles = indices.size() / 3;

        createMeshTime = (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
    }
    else {
        QuadsGenerator::BufferSizes bufferSizes = { 0 };
        Buffer<QuadsGenerator::BufferSizes> sizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, 1, &bufferSizes);

        startTime = glfwGetTime();
        std::string quadProxiesFileName = DATA_PATH + "quads.bin";
        auto quadProxiesData = FileIO::loadBinaryFile(quadProxiesFileName);

        // first uint in the file is the number of proxies
        unsigned int numProxies = *reinterpret_cast<unsigned int*>(quadProxiesData.data());
        unsigned int bufferOffset = sizeof(unsigned int);

        mesh = new Mesh({
            .numVertices = numProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
            .numIndices = numProxies * NUM_SUB_QUADS * 2 * 3,
            .material = new QuadMaterial({ .baseColorTexture = &colorTexture }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        });

        // next batch is the normalSphericals
        auto normalSphericalsPtr = reinterpret_cast<unsigned int*>(quadProxiesData.data() + bufferOffset);
        inputNormalSphericalsBuffer.bind();
        inputNormalSphericalsBuffer.setData(numProxies, normalSphericalsPtr);
        bufferOffset += numProxies * sizeof(unsigned int);

        // next batch is the depths
        auto depthsPtr = reinterpret_cast<float*>(quadProxiesData.data() + bufferOffset);
        inputDepthsBuffer.bind();
        inputDepthsBuffer.setData(numProxies, depthsPtr);
        bufferOffset += numProxies * sizeof(float);

        // next batch is the uvs
        auto uvsPtr = reinterpret_cast<unsigned int*>(quadProxiesData.data() + bufferOffset);
        inputUVsBuffer.bind();
        inputUVsBuffer.setData(numProxies, uvsPtr);
        bufferOffset += numProxies * sizeof(unsigned int);

        // last batch is the offsets
        auto offsetSizeFlattenedsPtr = reinterpret_cast<unsigned int*>(quadProxiesData.data() + bufferOffset);
        inputOffsetSizeFlattenedsBuffer.bind();
        inputOffsetSizeFlattenedsBuffer.setData(numProxies, offsetSizeFlattenedsPtr);

        glm::uvec2 depthBufferSize = 4u * windowSize;

        loadFromFilesTime = (glfwGetTime() - startTime) * MILLISECONDS_IN_SECOND;
        startTime = glfwGetTime();

        meshFromQuads.createMeshFromProxies(
            numProxies, depthBufferSize, remoteCamera,
            inputNormalSphericalsBuffer, inputDepthsBuffer, inputUVsBuffer, inputOffsetSizeFlattenedsBuffer,
            sizesBuffer, *mesh
        );

        createMeshTime = meshFromQuads.stats.timeToCreateMeshMs;

        sizesBuffer.bind();
        sizesBuffer.getSubData(0, 1, &bufferSizes);

        totalTriangles = bufferSizes.numIndices / 3;
        totalProxies = numProxies;
        totalDepthOffsets = 0;
    }

    Node node(mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Node nodeWireframe(mesh);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = new QuadMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    scene.addChildNode(&nodeWireframe);

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";

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
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

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

            if (totalProxies != -1 && totalDepthOffsets != -1) {
                float proxySizeMb = static_cast<float>(totalProxies * sizeof(QuadMapDataPacked)) / BYTES_IN_MB;
                float depthOffsetSizeMb = static_cast<float>(totalDepthOffsets * sizeof(uint16_t)) / BYTES_IN_MB;
                ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", totalProxies, proxySizeMb);
                ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", totalDepthOffsets, depthOffsetSizeMb);
            }

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

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to load data: %.3f ms", loadFromFilesTime);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to create mesh: %.3f ms", createMeshTime);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &nodeWireframe.visible);

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

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

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        renderer.drawToScreen(toneMapShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
