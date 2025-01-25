#include <iostream>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Recorder.h>
#include <Animator.h>

#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <shaders_common.h>

const std::string DATA_PATH = "../simulator/";

const std::vector<glm::vec4> colors = {
    glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
    glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
    glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "Depth Peeling Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'V', "vsync"}, true);
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<int> maxLayersIn(parser, "layers", "Max layers", {'n', "max-layers"}, 4);
    args::Flag loadProxies(parser, "load-proxies", "Load proxies from quads.bin.lz4", {'m', "load-proxies"});
    args::Flag disableWideFov(parser, "disable-wide-fov", "Disable wide fov view", {'W', "disable-wide-fov"});
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
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);

    std::string sceneFile = args::get(sceneFileIn);

    int maxLayers = args::get(maxLayersIn);
    int maxViews = !disableWideFov ? maxLayers + 1 : maxLayers;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();
    glm::uvec2 halfWindowSize = windowSize / 2u;

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    remoteCamera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    remoteCamera.updateViewMatrix();

    PerspectiveCamera remoteCameraWideFov(windowSize.x, windowSize.y);
    remoteCameraWideFov.setFovyDegrees(120.0f);
    remoteCameraWideFov.setViewMatrix(remoteCamera.getViewMatrix());

    // shaders
    ToneMapShader toneMapShader;

    Recorder recorder(renderer, toneMapShader, config.targetFramerate);

    MeshFromQuads meshFromQuads(halfWindowSize);

    std::vector<Texture> colorTextures; colorTextures.reserve(maxViews);
    TextureFileCreateParams params = {
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .flipVertically = true
    };
    for (int view = 0; view < maxViews; view++) {
        std::string colorFileName = DATA_PATH + "color" + std::to_string(view) + ".png";
        params.path = colorFileName;
        colorTextures.emplace_back(params);
    }

    std::vector<Mesh*> meshes(maxViews);
    std::vector<Node*> nodes(maxViews);
    std::vector<Node*> nodeWireframes(maxViews);

    unsigned int totalTriangles = -1;
    unsigned int totalProxies = -1;
    unsigned int totalDepthOffsets = -1;

    unsigned int numBytesProxies = 0;
    unsigned int numBytesDepthOffsets = 0;

    double startTime = window->getTime();
    double loadFromFilesTime = 0.0;
    double createMeshTime = 0.0;
    if (!args::get(loadProxies)) {
        for (int view = 0; view < maxViews; view++) {
            startTime = window->getTime();
            std::string verticesFileName = DATA_PATH + "vertices" + std::to_string(view) + ".bin.lz4";
            std::string indicesFileName = DATA_PATH + "indices" + std::to_string(view) + ".bin.lz4";

            auto vertexData = FileIO::loadBinaryFile(verticesFileName);
            auto indexData = FileIO::loadBinaryFile(indicesFileName);

            loadFromFilesTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
            startTime = window->getTime();

            std::vector<Vertex> vertices(vertexData.size() / sizeof(Vertex));
            std::memcpy(vertices.data(), vertexData.data(), vertexData.size());

            std::vector<unsigned int> indices(indexData.size() / sizeof(unsigned int));
            std::memcpy(indices.data(), indexData.data(), indexData.size());

            meshes[view] = new Mesh({
                .vertices = vertices,
                .indices = indices,
                .material = new QuadMaterial({ .baseColorTexture = &colorTextures[view] }),
            });

            totalTriangles += indices.size() / 3;

            createMeshTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;
        }
    }
    else {
        unsigned int maxProxies = windowSize.x * windowSize.y * NUM_SUB_QUADS;
        QuadBuffers quadBuffers(maxProxies);

        const glm::uvec2 depthBufferSize = 2u * windowSize;
        DepthOffsets depthOffsets(depthBufferSize);

        unsigned int numBytes;
        for (int view = 0; view < maxViews; view++) {
            startTime = window->getTime();

            // load proxies
            std::string quadProxiesFileName = DATA_PATH + "quads" + std::to_string(view) + ".bin.lz4";
            unsigned int numProxies = quadBuffers.loadFromFile(quadProxiesFileName, &numBytes);
            numBytesProxies += numBytes;
            // load depth offsets
            std::string depthOffsetsFileName = DATA_PATH + "depthOffsets" + std::to_string(view) + ".bin.lz4";
            unsigned int numDepthOffsets = depthOffsets.loadFromFile(depthOffsetsFileName, &numBytes);
            numBytesDepthOffsets += numBytes;

            meshes[view] = new Mesh({
                .numVertices = numProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
                .numIndices = numProxies * NUM_SUB_QUADS * 2 * 3,
                .material = new QuadMaterial({ .baseColorTexture = &colorTextures[view] }),
                .usage = GL_DYNAMIC_DRAW,
                .indirectDraw = true
            });
            loadFromFilesTime += (window->getTime() - startTime) * MILLISECONDS_IN_SECOND;

            glm::uvec2 gBufferSize = glm::uvec2(colorTextures[view].width, colorTextures[view].height) / 2u;
            if (view == maxViews - 1) {
                gBufferSize /= 2u;
            }

            startTime = window->getTime();
            auto& cameraToUse = (!disableWideFov && view == maxViews - 1) ? remoteCameraWideFov : remoteCamera;
            meshFromQuads.appendProxies(
                gBufferSize,
                numProxies, quadBuffers
            );
            meshFromQuads.createMeshFromProxies(
                gBufferSize,
                numProxies, depthOffsets,
                cameraToUse,
                *meshes[view]
            );
            createMeshTime += meshFromQuads.stats.timeToCreateMeshMs;

            auto meshBufferSizes = meshFromQuads.getBufferSizes();

            totalTriangles += meshBufferSizes.numIndices / 3;
            totalProxies += numProxies;
            totalDepthOffsets = numDepthOffsets;
        }
    }

    for (int view = 0; view < maxViews; view++) {
        nodes[view] = new Node(meshes[view]);
        nodes[view]->frustumCulled = false;
        scene.addChildNode(nodes[view]);

        nodeWireframes[view] = new Node(meshes[view]);
        nodeWireframes[view]->frustumCulled = false;
        nodeWireframes[view]->wireframe = true;
        nodeWireframes[view]->visible = false;
        nodeWireframes[view]->overrideMaterial = new QuadMaterial({ .baseColor = colors[view] });
        scene.addChildNode(nodeWireframes[view]);
    }

    bool* showLayers = new bool[maxViews];
    for (int i = 0; i < maxViews; ++i) {
        showLayers[i] = true;
    }

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

            bool showWireframe = nodeWireframes[0]->visible;
            ImGui::Checkbox("Show Wireframe", &showWireframe);
            for (int i = 0; i < maxViews; i++) {
                nodeWireframes[i]->visible = showWireframe;
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

        for (int i = 0; i < maxViews; i++) {
            nodes[i]->visible = showLayers[i];
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
