#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <PostProcessing/Tonemapper.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/FrameCaptureWindow.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Receivers/QuadStreamReceiver.h>

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
    config.title = "QuadStream Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<int> maxAdditionalViewsIn(parser, "maxViews", "Max views", {'l', "num-views"}, 8);
    args::Flag disableWideFov(parser, "disable-wide-fov", "Disable wide fov view", {'W', "disable-wide-fov"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
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

    Path dataPath = Path(args::get(dataPathIn));
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);

    // Post processing
    Tonemapper tonemapper;

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, tonemapper, dataPath, config.targetFramerate);

    QuadSet quadSet(windowSize);
    QuadStreamReceiver quadstreamReceiver(quadSet, maxViews);

    // Create node and wireframe node
    std::vector<Node> nodes(maxViews);
    std::vector<Node> nodeWireframes(maxViews);
    // Add in reverse order to have correct layering
    for (int view = maxViews - 1; view >= 0; view--) {
        nodes[view].addEntity(&quadstreamReceiver.getMesh(view));
        nodes[view].frustumCulled = false;
        scene.addChildNode(&nodes[view]);

        nodeWireframes[view].addEntity(&quadstreamReceiver.getMesh(view));
        nodeWireframes[view].frustumCulled = false;
        nodeWireframes[view].wireframe = true;
        nodeWireframes[view].visible = false;
        nodeWireframes[view].overrideMaterial = new QuadMaterial({ .baseColor = colors[view % colors.size()] });
        scene.addChildNode(&nodeWireframes[view]);
    }

    // Initial load
    quadstreamReceiver.loadFromFiles(dataPath);
    quadstreamReceiver.copyPoseToCamera(camera);

    bool restrictMovementToViewBox = true;

    bool* showViews = new bool[maxViews];
    for (int i = 0; i < maxViews; i++) {
        showViews[i] = true;
    }
    bool showWireframe = false;

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, glm::uvec2(430, 270), outputPath);
    CameraHeader cameraHeader(camera);
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = true;

        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &frameRateWindow.visible);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Frame Capture", 0, &frameCaptureWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (quadstreamReceiver.stats.totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", quadstreamReceiver.stats.totalTriangles);
            else if (quadstreamReceiver.stats.totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", quadstreamReceiver.stats.totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", quadstreamReceiver.stats.totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               quadstreamReceiver.stats.sizes.numQuads,
                               quadstreamReceiver.stats.sizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quadstreamReceiver.stats.sizes.numDepthOffsets,
                               quadstreamReceiver.stats.sizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to load data: %.3f ms", quadstreamReceiver.stats.loadTimeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decompress data: %.3f ms", quadstreamReceiver.stats.decompressTimeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy data to GPU: %.3f ms", quadstreamReceiver.stats.transferTimeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to create mesh: %.3f ms", quadstreamReceiver.stats.createMeshTimeMs);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);

            ImGui::Separator();
            if (ImGui::Button("Reload Proxies", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                quadstreamReceiver.loadFromFiles(dataPath);
            }

            ImGui::Checkbox("Restrict Movement to View Box", &restrictMovementToViewBox);

            ImGui::Separator();

            const int columns = 4;
            for (int view = 0; view < maxViews; view++) {
                ImGui::Checkbox(("Show View " + std::to_string(view)).c_str(), &showViews[view]);
                if ((view + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();
        }

        frameCaptureWindow.draw(now, dt);
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

        for (int i = 0; i < maxViews; i++) {
            nodes[i].visible = showViews[i];
            nodeWireframes[i].visible = showWireframe && showViews[i];
        }

        if (restrictMovementToViewBox) {
            glm::vec3 remotePosition = quadstreamReceiver.getRemoteCamera().getPosition();
            glm::vec3 position = camera.getPosition();
            // Restrict camera position to be inside position±viewBoxSize
            position.x = glm::clamp(position.x, remotePosition.x - quadstreamReceiver.viewBoxSize/2, remotePosition.x + quadstreamReceiver.viewBoxSize/2);
            position.y = glm::clamp(position.y, remotePosition.y - quadstreamReceiver.viewBoxSize/2, remotePosition.y + quadstreamReceiver.viewBoxSize/2);
            position.z = glm::clamp(position.z, remotePosition.z - quadstreamReceiver.viewBoxSize/2, remotePosition.z + quadstreamReceiver.viewBoxSize/2);
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        tonemapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
