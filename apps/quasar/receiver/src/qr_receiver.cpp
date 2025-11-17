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
#include <UI/TexturePreviewWindow.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Receivers/QUASARReceiver.h>
#include <Streamers/PoseStreamer.h>

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
    args::Flag loadFromDisk(parser, "load-from-disk", "Load data from disk", {'L', "load-from-disk"}, false);
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Path to output files", {'O', "output-path"}, ".");
    args::ValueFlag<std::string> videoURLIn(parser, "video", "URL to recv video", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> proxiesURLIn(parser, "proxies", "URL to recv quad proxy metadata", {'e', "proxies-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "URL to recv camera pose", {'p', "pose-url"}, "127.0.0.1:54321");
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

    Path dataPath = Path(args::get(dataPathIn));
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    std::string videoURL = !loadFromDisk ? args::get(videoURLIn) : "";
    std::string proxiesURL = !loadFromDisk ? args::get(proxiesURLIn) : "";
    std::string poseURL = !loadFromDisk ? args::get(poseURLIn) : "";

    int maxHidLayers = args::get(maxHiddenLayersIn);
    int maxLayers = maxHidLayers + 2;

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
    QUASARReceiver quasarReceiver(quadSet, maxLayers, videoURL, proxiesURL);

    PoseStreamer poseStreamer(&camera, poseURL);

    // Create node and wireframe node
    std::vector<Node> refNodes(maxLayers);
    std::vector<Node> refNodeWireframes(maxLayers);
    // Add in reverse order to have correct layering
    for (int layer = maxLayers - 1; layer >= 0; layer--) {
        refNodes[layer].addEntity(&quasarReceiver.getMesh(layer));
        refNodes[layer].frustumCulled = false;
        scene.addChildNode(&refNodes[layer]);

        refNodeWireframes[layer].addEntity(&quasarReceiver.getMesh(layer));
        refNodeWireframes[layer].frustumCulled = false;
        refNodeWireframes[layer].wireframe = true;
        refNodeWireframes[layer].visible = false;
        refNodeWireframes[layer].overrideMaterial = new QuadMaterial({ .baseColor = colors[layer % colors.size()] });
        scene.addChildNode(&refNodeWireframes[layer]);
    }

    Node resNode(&quasarReceiver.getResidualMesh());
    resNode.frustumCulled = false;
    scene.addChildNode(&resNode);

    QuadMaterial resNodeWireframeMaterial({ .baseColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f) });
    Node resNodeWireframe(&quasarReceiver.getResidualMesh());
    resNodeWireframe.frustumCulled = false;
    resNodeWireframe.wireframe = true;
    resNodeWireframe.visible = false;
    resNodeWireframe.overrideMaterial = &resNodeWireframeMaterial;
    scene.addChildNode(&resNodeWireframe);

    if (loadFromDisk) {
        // Initial load
        quasarReceiver.loadFromFiles(dataPath);
        quasarReceiver.copyPoseToCamera(camera);
    }

    bool restrictMovementToViewSphere = loadFromDisk;

    bool* showLayers = new bool[maxLayers];
    for (int i = 0; i < maxLayers; i++) {
        showLayers[i] = true;
    }
    bool showWireframe = false;

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, glm::uvec2(430, 270), outputPath);
    TexturePreviewWindow videoPreviewWindow("Video Texture", quasarReceiver.videoAtlasTexture, glm::uvec2(860, 860));
    TexturePreviewWindow alphaPreviewWindow("Alpha Texture", quasarReceiver.alphaAtlasTexture, glm::uvec2(860, 860));
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
            ImGui::MenuItem("Video Preview", 0, &videoPreviewWindow.visible);
            ImGui::MenuItem("Alpha Preview", 0, &alphaPreviewWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        frameCaptureWindow.draw(now, dt);
        videoPreviewWindow.draw(now, dt);
        alphaPreviewWindow.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (quasarReceiver.stats.totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", quasarReceiver.stats.totalTriangles);
            else if (quasarReceiver.stats.totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", quasarReceiver.stats.totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", quasarReceiver.stats.totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               quasarReceiver.stats.sizes.numQuads,
                               quasarReceiver.stats.sizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quasarReceiver.stats.sizes.numDepthOffsets,
                               quasarReceiver.stats.sizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Video Stats")) {
                ImGui::TextColored(ImVec4(1,0.5,0,1), "Frame Rate: %.1f FPS (%.3f ms/frame)",
                                                        quasarReceiver.videoAtlasTexture.getFrameRate(),
                                                        1000.0f / quasarReceiver.videoAtlasTexture.getFrameRate());
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms",
                                                        quasarReceiver.videoAtlasTexture.stats.receiveTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0.5,1), "Bitrate: %.3f Mbps",
                                                        quasarReceiver.videoAtlasTexture.stats.bitrateMbps);
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Proxy Stats")) {
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to load data: %.3f ms", quasarReceiver.stats.loadTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decompress data (async): %.3f ms", quasarReceiver.stats.decompressTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy data to GPU: %.3f ms", quasarReceiver.stats.transferTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to create mesh: %.3f ms", quasarReceiver.stats.createMeshTimeMs);
            }

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);

            if (loadFromDisk) {
                ImGui::Separator();
                if (ImGui::Button("Reload Proxies", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    quasarReceiver.loadFromFiles(dataPath);
                }
            }

            ImGui::Checkbox("Restrict Movement to View Sphere", &restrictMovementToViewSphere);

            ImGui::Separator();

            const int columns = 3;
            for (int layer = 0; layer < maxLayers; layer++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(layer)).c_str(), &showLayers[layer]);
                if ((layer + 1) % columns != 0) {
                    ImGui::SameLine();
                }
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

        // Send pose to streamer
        pose_id_t currPoseID = poseStreamer.sendPose();
        poseStreamer.removePosesLessThan(currPoseID);

        QuadFrame::FrameType frameType = quasarReceiver.recvData();
        if (frameType != QuadFrame::FrameType::NONE) {
            resNode.visible = frameType == QuadFrame::FrameType::RESIDUAL;
        }
        for (int i = 0; i < maxLayers; i++) {
            refNodes[i].visible = showLayers[i];
            refNodeWireframes[i].visible = showWireframe && showLayers[i];
        }
        resNodeWireframe.visible = resNode.visible && showWireframe;

        if (restrictMovementToViewSphere) {
            glm::vec3 remotePosition = quasarReceiver.getRemoteCamera().getPosition();
            glm::vec3 position = camera.getPosition();
            glm::vec3 direction = position - remotePosition;
            float distanceSquared = glm::dot(direction, direction);
            float radius = quasarReceiver.viewSphereDiameter / 2.0f;
            if (distanceSquared > radius * radius) {
                position = remotePosition + glm::normalize(direction) * radius;
            }
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        // Render generated meshes
        quasarReceiver.setDrawState(QuadMesh::DrawState::OPAQUE); // draw opaque quads first
        renderStats = renderer.drawObjects(scene, camera);
        quasarReceiver.setDrawState(QuadMesh::DrawState::TRANSPARENT); // then draw transparent quads
        renderStats += renderer.drawObjects(scene, camera, 0);

        // Render to screen
        tonemapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
