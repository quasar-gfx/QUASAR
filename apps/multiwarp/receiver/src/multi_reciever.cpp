
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

const std::vector<glm::vec3> offsets = {
    glm::vec3(-1.0f, +1.0f, -1.0f), // Top-left
    glm::vec3(+1.0f, +1.0f, -1.0f), // Top-right
    glm::vec3(+1.0f, -1.0f, -1.0f), // Bottom-right
    glm::vec3(-1.0f, -1.0f, -1.0f), // Bottom-left
    glm::vec3(-1.0f, +1.0f, +1.0f), // Top-left
    glm::vec3(+1.0f, +1.0f, +1.0f), // Top-right
    glm::vec3(+1.0f, -1.0f, +1.0f), // Bottom-right
    glm::vec3(-1.0f, -1.0f, +1.0f), // Bottom-left
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "Multi-Camera Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<int> maxAdditionalViewsIn(parser, "maxViews", "Max views", {'l', "num-views"}, 8);
    args::Flag disableWideFov(parser, "disable-wide-fov", "Disable wide fov view", {'W', "disable-wide-fov"});args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 60.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 120.0f);
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

    int maxAdditionalViews = args::get(maxAdditionalViewsIn);
    int maxViews = !disableWideFov ? maxAdditionalViews + 2 : maxAdditionalViews + 1;

    config.enableVSync = !args::get(novsync);

    Path dataPath = Path(args::get(dataPathIn)); dataPath.mkdirRecursive();

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    float viewBoxSize = 0.5f;

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);

    float remoteFOV = args::get(remoteFOVIn);
    std::vector<PerspectiveCamera> remoteCameras; remoteCameras.reserve(maxViews);
    for (int view = 0; view < maxViews; view++) {
        remoteCameras.emplace_back(windowSize.x, windowSize.y);
        remoteCameras[view].setFovyDegrees(remoteFOV);
    }
    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    remoteCameraCenter.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    remoteCameraCenter.updateViewMatrix();

    for (int view = 1; view < maxViews - 1; view++) {
        const glm::vec3& offset = offsets[view - 1];
        const glm::vec3& right = remoteCameraCenter.getRightVector();
        const glm::vec3& up = remoteCameraCenter.getUpVector();
        const glm::vec3& forward = remoteCameraCenter.getForwardVector();

        glm::vec3 worldOffset =
            right   * offset.x * viewBoxSize / 2.0f +
            up      * offset.y * viewBoxSize / 2.0f +
            forward * -offset.z * viewBoxSize / 2.0f;

        remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
        remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + worldOffset);
        remoteCameras[view].updateViewMatrix();
    }

    // Make last camera have a larger fov
    float remoteFOVWide = args::get(remoteFOVWideIn);
    remoteCameras[maxViews-1].setFovyDegrees(remoteFOVWide);
    remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());

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
        .magFilter = GL_LINEAR
    }, renderer, toneMapper, dataPath, config.targetFramerate);

    MeshFromQuads meshFromQuads(windowSize);

    std::vector<Texture> colorTextures; colorTextures.reserve(maxViews);
    TextureFileCreateParams params = {
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .flipVertically = true
    };
    for (int view = 0; view < maxViews; view++) {
        Path colorFileName = dataPath.appendToName("color" + std::to_string(view));
        params.path = colorFileName.withExtension(".jpg");
        colorTextures.emplace_back(params);
    }

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

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

    uint numBytes;
    for (int view = 0; view < maxViews; view++) {
        startTime = window->getTime();

        // Load proxies
        Path quadProxiesFileName = (dataPath / "quads").appendToName(std::to_string(view)).withExtension(".bin.zstd");
        uint numProxies = quadBuffers.loadFromFile(quadProxiesFileName, &numBytes);
        totalBytesProxies += numBytes;
        // Load depth offsets
        Path depthOffsetsFileName = (dataPath / "depthOffsets").appendToName(std::to_string(view)).withExtension(".bin.zstd");
        uint numDepthOffsets = depthOffsets.loadFromFile(depthOffsetsFileName, &numBytes);
        totalBytesDepthOffsets += numBytes;

        meshes[view] = new Mesh({
            .maxVertices = numProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
            .maxIndices = numProxies * NUM_SUB_QUADS * INDICES_IN_A_QUAD,
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .material = new QuadMaterial({ .baseColorTexture = &colorTextures[view] }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        });
        loadFromFilesTime += timeutils::secondsToMillis(window->getTime() - startTime);

        const glm::vec2 gBufferSize = glm::vec2(colorTextures[view].width, colorTextures[view].height);

        startTime = window->getTime();
        meshFromQuads.appendQuads(
            gBufferSize,
            numProxies, quadBuffers
        );
        meshFromQuads.createMeshFromProxies(
            gBufferSize,
            numProxies, depthOffsets,
            remoteCameras[view],
            *meshes[view]
        );
        createMeshTime += meshFromQuads.stats.timeToCreateMeshMs;

        auto meshBufferSizes = meshFromQuads.getBufferSizes();

        totalTriangles += meshBufferSizes.numIndices / 3;
        totalProxies += numProxies;
        totalDepthOffsets = numDepthOffsets;
    }

    for (int view = 0; view < maxViews; view++) {
        nodes[view] = new Node(meshes[view]);
        nodes[view]->frustumCulled = false;
        scene.addChildNode(nodes[view]);

        // Primary view color is yellow
        glm::vec4 color = (view == 0) ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) :
                glm::vec4(fmod(view * 0.6180339887f, 1.0f),
                            fmod(view * 0.9f, 1.0f),
                            fmod(view * 0.5f, 1.0f),
                            1.0f);

        nodeWireframes[view] = new Node(meshes[view]);
        nodeWireframes[view]->frustumCulled = false;
        nodeWireframes[view]->wireframe = true;
        nodeWireframes[view]->visible = false;
        nodeWireframes[view]->overrideMaterial = new QuadMaterial({ .baseColor = color });
        scene.addChildNode(nodeWireframes[view]);
    }

    bool* showViews = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showViews[i] = true;
    }

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

            float proxySizeMB = static_cast<float>(totalBytesProxies) / BYTES_IN_MB;
            float depthOffsetSizeMB = static_cast<float>(totalBytesDepthOffsets) / BYTES_IN_MB;
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

            bool showWireframe = nodeWireframes[0]->visible;
            ImGui::Checkbox("Show Wireframe", &showWireframe);
            for (int i = 0; i < maxViews; i++) {
                nodeWireframes[i]->visible = showWireframe;
            }

            ImGui::Separator();

            const int columns = 3;
            for (int i = 0; i < maxViews; i++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(i)).c_str(), &showViews[i]);
                if ((i + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

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

        for (int i = 0; i < maxViews; i++) {
            bool showLayer = showViews[i];

            nodes[i]->visible = showLayer;
        }

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        toneMapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
