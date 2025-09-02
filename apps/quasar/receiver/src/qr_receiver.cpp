#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <PostProcessing/ToneMapper.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Receivers/QUASARReceiver.h>

using namespace quasar;

const std::vector<glm::vec4> colors = {
    glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
    glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
    glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
    glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "QUASAR Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    args::Flag disableWideFov(parser, "disable-wide-fov", "Disable wide fov view", {'W', "disable-wide-fov"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 60.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'R', "remote-fov-wide"}, 120.0f);
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

    if (verbose) spdlog::set_level(spdlog::level::debug);

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);

    Path dataPath = Path(args::get(dataPathIn)); dataPath.mkdirRecursive();

    int maxHidLayers = args::get(maxHiddenLayersIn);
    int maxLayers = !disableWideFov ? maxHidLayers + 2 : maxHidLayers;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);

    // Post processing
    ToneMapper toneMapper(false);

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

    QuadSet quadSet(windowSize);
    float remoteFOV = args::get(remoteFOVIn);
    float remoteFOVWide = args::get(remoteFOVWideIn);
    QUASARReceiver quasarReceiver(quadSet, maxLayers, remoteFOV, remoteFOVWide);

    // Create node and wireframe node
    std::vector<Node> nodes(maxLayers);
    std::vector<Node> nodeWireframes(maxLayers);
    for (int layer = 0; layer < maxLayers; layer++) {
        nodes[layer].setEntity(&quasarReceiver.getMesh(layer));
        nodes[layer].frustumCulled = false;
        scene.addChildNode(&nodes[layer]);

        nodeWireframes[layer].setEntity(&quasarReceiver.getMesh(layer));
        nodeWireframes[layer].frustumCulled = false;
        nodeWireframes[layer].wireframe = true;
        nodeWireframes[layer].visible = false;
        nodeWireframes[layer].overrideMaterial = new QuadMaterial({ .baseColor = colors[layer % colors.size()] });
        scene.addChildNode(&nodeWireframes[layer]);
    }

    // Initial load
    quasarReceiver.loadFromFiles(dataPath);
    quasarReceiver.copyPoseToCamera(camera);

    bool* showLayers = new bool[maxLayers];
    for (int i = 0; i < maxLayers; ++i) {
        showLayers[i] = true;
    }
    bool showWireframe = false;

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showFrameCaptureWindow = false;
        static bool writeToHDR = false;
        static char fileNameBase[256] = "screenshot";

        ImGui::NewFrame();

        ImGuiWindowFlags flags = 0;
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

            if (quasarReceiver.stats.totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %d", quasarReceiver.stats.totalTriangles);
            else if (quasarReceiver.stats.totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %d", quasarReceiver.stats.totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %d", quasarReceiver.stats.totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %d", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               quasarReceiver.stats.sizes.numQuads,
                               quasarReceiver.stats.sizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quasarReceiver.stats.sizes.numDepthOffsets,
                               quasarReceiver.stats.sizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

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

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to load data: %.3f ms", quasarReceiver.stats.timeToLoadMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decompress data: %.3f ms", quasarReceiver.stats.timeToDecompressMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy data to GPU: %.3f ms", quasarReceiver.stats.timeToTransferMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to create mesh: %.3f ms", quasarReceiver.stats.timeToCreateMeshMs);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);

            ImGui::Separator();

            const int columns = 3;
            for (int i = 0; i < maxLayers; i++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(i)).c_str(), &showLayers[i]);
                if ((i + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::Separator();

            if (ImGui::Button("Reload Data", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                quasarReceiver.loadFromFiles(dataPath);
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

            ImGui::Checkbox("Save as HDR", &writeToHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(filename, writeToHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);
        camera.setAspect(windowSize);
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

        for (int i = 0; i < maxLayers; ++i) {
            nodes[i].visible = showLayers[i];
            nodeWireframes[i].visible = showWireframe && showLayers[i];
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
