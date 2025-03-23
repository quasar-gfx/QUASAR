
#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <PostProcessing/ToneMapper.h>

#include <Recorder.h>
#include <Animator.h>

#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <shaders_common.h>

const std::string DATA_PATH = "../simulator/";

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag loadProxies(parser, "load-proxies", "Load proxies from quads.bin.zstd", {'m', "load-proxies"});
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

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    }

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);

    std::string sceneFile = args::get(sceneFileIn);
    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    remoteCamera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    remoteCamera.updateViewMatrix();

    // post processing
    ToneMapper toneMapper;

    Recorder recorder(renderer, toneMapper, config.targetFramerate);

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

    unsigned int numBytesProxies = 0;
    unsigned int numBytesDepthOffsets = 0;

    double startTime = window->getTime();
    double loadFromFilesTime = 0.0;
    double createMeshTime = 0.0;
    if (!args::get(loadProxies)) {
        std::string verticesFileName = DATA_PATH + "vertices.bin.zstd";
        std::string indicesFileName = DATA_PATH + "indices.bin.zstd";

        auto vertexData = FileIO::loadBinaryFile(verticesFileName);
        auto indexData = FileIO::loadBinaryFile(indicesFileName);

        loadFromFilesTime = window->getTime() - startTime;
        startTime = window->getTime();

        std::vector<Vertex> vertices(vertexData.size() / sizeof(Vertex));
        std::memcpy(vertices.data(), vertexData.data(), vertexData.size());

        std::vector<unsigned int> indices(indexData.size() / sizeof(unsigned int));
        std::memcpy(indices.data(), indexData.data(), indexData.size());

        mesh = new Mesh({
            .verticesData = vertices.data(),
            .verticesSize = vertices.size(),
            .indicesData = indices.data(),
            .indicesSize = indices.size(),
            .material = new QuadMaterial({ .baseColorTexture = &colorTexture }),
        });

        totalTriangles = indices.size() / 3;

        createMeshTime = timeutils::secondsToMillis(window->getTime() - startTime);
    }
    else {
        const glm::uvec2 halfWindowSize = windowSize / 2u;
        unsigned int maxProxies = halfWindowSize.x * halfWindowSize.y * NUM_SUB_QUADS;
        QuadBuffers quadBuffers(maxProxies);

        const glm::uvec2 depthBufferSize = 2u * halfWindowSize;
        DepthOffsets depthOffsets(depthBufferSize);

        startTime = window->getTime();
        // load proxies
        unsigned int numProxies = quadBuffers.loadFromFile(DATA_PATH + "quads.bin.zstd", &numBytesProxies);
        // load depth offsets
        unsigned int numDepthOffsets = depthOffsets.loadFromFile(DATA_PATH + "depthOffsets.bin.zstd", &numBytesDepthOffsets);

        mesh = new Mesh({
            .maxVertices = numProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
            .maxIndices = numProxies * NUM_SUB_QUADS * 2 * 3,
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .material = new QuadMaterial({ .baseColorTexture = &colorTexture }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        });
        loadFromFilesTime = timeutils::secondsToMillis(window->getTime() - startTime);

        startTime = window->getTime();
        meshFromQuads.appendProxies(
            glm::vec2(halfWindowSize.x, halfWindowSize.y),
            numProxies,
            quadBuffers
        );
        meshFromQuads.createMeshFromProxies(
            glm::vec2(halfWindowSize.x, halfWindowSize.y),
            numProxies, depthOffsets,
            remoteCamera,
            *mesh
        );

        createMeshTime = meshFromQuads.stats.timeToCreateMeshMs;

        auto meshBufferSizes = meshFromQuads.getBufferSizes();

        totalTriangles = meshBufferSizes.numIndices / 3;
        totalProxies = numProxies;
        totalDepthOffsets = numDepthOffsets;
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

            float proxySizeMb = static_cast<float>(numBytesProxies) / BYTES_IN_MB;
            float depthOffsetSizeMb = static_cast<float>(numBytesDepthOffsets) / BYTES_IN_MB;
            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", totalProxies, proxySizeMb);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", totalDepthOffsets, depthOffsetSizeMb);

            ImGui::Separator();

            glm::vec3 position = camera.getPosition();
            if (ImGui::DragFloat3("Camera Position", (float*)&position, 0.01f)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::DragFloat3("Camera Rotation", (float*)&rotation, 0.1f)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::DragFloat("Movement Speed", &camera.movementSpeed, 0.05f, 0.1f, 20.0f);

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
        auto keys = window->getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window->close();
        }
        auto scroll = window->getScrollOffset();
        camera.processScroll(scroll.y);

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        toneMapper.drawToScreen(renderer);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
