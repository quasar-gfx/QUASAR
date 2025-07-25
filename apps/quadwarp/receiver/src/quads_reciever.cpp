
#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <PostProcessing/ToneMapper.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <shaders_common.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 60.0f);
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

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);

    Path dataPath = Path(args::get(dataPathIn));

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

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);

    // Post processing
    ToneMapper toneMapper;
    toneMapper.enableToneMapping(false);

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, toneMapper, dataPath, config.targetFramerate);

    MeshFromQuads meshFromQuads(windowSize);

    std::string colorFileName = dataPath / "color.jpg";
    Texture colorTexture = Texture({
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .flipVertically = true,
        .path = colorFileName,
    });

    Mesh* mesh;

    uint totalTriangles = -1;
    uint totalProxies = -1;
    uint totalDepthOffsets = -1;

    uint totalBytesProxies = 0;
    uint totalBytesDepthOffsets = 0;

    double startTime = window->getTime();
    double loadFromFilesTime = 0.0;
    double createMeshTime = 0.0;

    uint maxProxies = windowSize.x * windowSize.y * NUM_SUB_QUADS;
    QuadBuffers quadBuffers(maxProxies);

    const glm::uvec2 depthOffsetBufferSize = 2u * windowSize;
    DepthOffsets depthOffsets(depthOffsetBufferSize);

    startTime = window->getTime();
    // Load proxies
    uint numProxies = quadBuffers.loadFromFile(dataPath / "quads.bin.zstd", &totalBytesProxies);
    // Load depth offsets
    uint numDepthOffsets = depthOffsets.loadFromFile(dataPath / "depthOffsets.bin.zstd", &totalBytesDepthOffsets);

    mesh = new Mesh({
        .maxVertices = numProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
        .maxIndices = numProxies * NUM_SUB_QUADS * INDICES_IN_A_QUAD,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = new QuadMaterial({ .baseColorTexture = &colorTexture ,}),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    });
    loadFromFilesTime = timeutils::secondsToMillis(window->getTime() - startTime);

    const glm::vec2 gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);

    startTime = window->getTime();
    meshFromQuads.appendQuads(
        gBufferSize,
        numProxies,
        quadBuffers
    );
    meshFromQuads.createMeshFromProxies(
        gBufferSize,
        numProxies, depthOffsets,
        remoteCamera,
        *mesh
    );

    createMeshTime = meshFromQuads.stats.timeToCreateMeshMs;

    auto meshBufferSizes = meshFromQuads.getBufferSizes();

    totalTriangles = meshBufferSizes.numIndices / 3;
    totalProxies = numProxies;
    totalDepthOffsets = numDepthOffsets;

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
        static bool showFrameCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";

        ImGui::NewFrame();

        uint flags = 0;
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
            ImGui::MenuItem("Frame Capture", 0, &showFrameCaptureWindow);
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

            float proxySizeMB = static_cast<float>(totalBytesProxies) / BYTES_PER_MEGABYTE;
            float depthOffsetSizeMB = static_cast<float>(totalBytesDepthOffsets) / BYTES_PER_MEGABYTE;
            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", totalProxies, proxySizeMB);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", totalDepthOffsets, depthOffsetSizeMB);

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

        if (showFrameCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showFrameCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            Path filename = (dataPath / fileNameBase).appendToName("." + time);

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(filename, saveAsHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    app.onRender([&](double now, double dt) {
        // Handle mouse input
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

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        toneMapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
