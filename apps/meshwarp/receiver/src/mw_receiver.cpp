#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <PostProcessing/Tonemapper.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Receivers/MeshWarpReceiver.h>
#include <Streamers/PoseStreamer.h>

using namespace quasar;

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag loadFromDisk(parser, "load-from-disk", "Load data from disk", {'L', "load-from-disk"}, false);
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<uint> vertexGroupSizeIn(parser, "vertex", "Size of vertex grouping", {'g', "vertex-group-size"}, 1);
    args::ValueFlag<uint> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote field of view", {'f', "remote-fov"}, 60.0f);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<std::string> videoURLIn(parser, "video", "URL to recv video", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "URL to recv depth", {'e', "depth-url"}, "127.0.0.1:65432");
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

    // Parse remote size
    std::string rsizeStr = args::get(resIn);
    pos = rsizeStr.find('x');
    glm::uvec2 remoteWindowSize = glm::uvec2(std::stoi(rsizeStr.substr(0, pos)), std::stoi(rsizeStr.substr(pos + 1)));

    config.enableVSync = !args::get(novsync);

    Path dataPath = Path(args::get(dataPathIn));
    std::string videoURL = !loadFromDisk ? args::get(videoURLIn) : "";
    std::string depthURL = !loadFromDisk ? args::get(depthURLIn) : "";
    std::string poseURL = !loadFromDisk ? args::get(poseURLIn) : "";

    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();

    uint vertexGroupSize = args::get(vertexGroupSizeIn);
    uint depthFactor = args::get(depthFactorIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    RenderState renderState = RenderState::MESH;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);

    // Post processing
    Tonemapper tonemapper(false);

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
    }, renderer, tonemapper, outputPath, config.targetFramerate);

    float remoteFOV = args::get(remoteFOVIn);
    MeshWarpReceiver meshWarpReceiver(remoteWindowSize, depthFactor, vertexGroupSize, remoteFOV, videoURL, depthURL);
    PoseStreamer poseStreamer(&meshWarpReceiver.getRemoteCamera(), poseURL);

    Node node(&meshWarpReceiver.getMesh());
    node.frustumCulled = false;
    node.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    scene.addChildNode(&node);

    UnlitMaterial wireframeMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    Node nodeWireframe(&meshWarpReceiver.getMesh());
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = &wireframeMaterial;
    scene.addChildNode(&nodeWireframe);

    if (loadFromDisk) {
        // Initial load
        meshWarpReceiver.loadFromFiles(dataPath);
        float fov = camera.getFovyDegrees();
        meshWarpReceiver.copyPoseToCamera(camera);
        camera.setFovyDegrees(fov);
    }

    Shader videoShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_SHOW_TEXTURE_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_SHOW_TEXTURE_FRAG_len,
    });

    double elapsedTimeColor, elapsedTimeDepth;
    bool mwEnabled = true;

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showFrameCaptureWindow = false;
        static bool writeToHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showVideoPreview = true;

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
            ImGui::MenuItem("Video Preview", 0, &showVideoPreview);
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
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

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

            if (ImGui::CollapsingHeader("Background Settings")) {
                if (ImGui::Button("Change Background Color", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    ImGui::OpenPopup("Background Color Popup");
                }
                if (ImGui::BeginPopup("Background Color Popup")) {
                    ImGui::ColorPicker3("Background Color", (float*)&scene.backgroundColor);
                    ImGui::EndPopup();
                }
            }

            ImGui::Separator();

            ImGui::Text("Remote Pose ID: RGB (%d), D (%d)", meshWarpReceiver.poseIdColor, meshWarpReceiver.poseIdDepth);

            glm::mat4 pose = glm::inverse(meshWarpReceiver.depthFramePose.mono.view);
            glm::vec3 skew, scale;
            glm::quat rotationQuat;
            glm::vec3 remotePosition;
            glm::vec4 perspective;
            glm::decompose(pose, scale, rotationQuat, remotePosition, skew, perspective);
            glm::vec3 remoteRotation = glm::degrees(glm::eulerAngles(rotationQuat));
            ImGui::BeginDisabled();
            ImGui::InputFloat3("Remote Position", (float*)&remotePosition);
            ImGui::InputFloat3("Remote Rotation", (float*)&remoteRotation);
            ImGui::EndDisabled();

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Depth URL: %s", depthURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f FPS), D (%.1f FPS)",
                                                    meshWarpReceiver.videoTexture.getFrameRate(),
                                                    meshWarpReceiver.depthTexture.getFrameRate());
            ImGui::TextColored(ImVec4(1,0.5,0,1), "E2E Latency: RGB (%.3f ms), D (%.3f ms)", elapsedTimeColor, elapsedTimeDepth);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms",
                                                    meshWarpReceiver.videoTexture.stats.receiveTimeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: RGB (%.3f Mbps), D (%.3f Mbps)",
                                                    meshWarpReceiver.videoTexture.stats.bitrateMbps,
                                                    meshWarpReceiver.depthTexture.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::Checkbox("Mesh Warp Enabled", &mwEnabled);

            ImGui::Checkbox("Sync Color and Depth", &meshWarpReceiver.sync);

            ImGui::Separator();

            ImGui::DragFloat("Remote FOV", &remoteFOV, 0.5f, 60.0f, 170.0f);

            ImGui::Separator();

            ImGui::RadioButton("Render Mesh", (int*)&renderState, 0);
            ImGui::RadioButton("Render Point Cloud", (int*)&renderState, 1);
            ImGui::RadioButton("Render Wireframe", (int*)&renderState, 2);

            ImGui::End();
        }

        flags = ImGuiWindowFlags_AlwaysAutoResize;
        if (showVideoPreview) {
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - windowSize.x / 4 - 60, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Video Texture", &showVideoPreview, flags);
            ImGui::Image((void*)(intptr_t)(meshWarpReceiver.videoTexture),
                         ImVec2(windowSize.x / 4, windowSize.y / 4), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        if (showFrameCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showFrameCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            Path filename = (outputPath / fileNameBase).appendToName("." + time);

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

        // Copy camera pose to remote camera
        auto& remoteCamera = meshWarpReceiver.getRemoteCamera();
        remoteCamera.setFovyDegrees(remoteFOV);
        remoteCamera.setViewMatrix(camera.getViewMatrix());

        // Send pose to streamer
        poseStreamer.sendPose();
        meshWarpReceiver.recvData(poseStreamer, elapsedTimeColor, elapsedTimeDepth);
        if (!mwEnabled) {
            videoShader.bind();
            videoShader.setTexture("tex", meshWarpReceiver.videoTexture, 5);
            renderStats = renderer.drawToScreen(videoShader);
            return;
        }
        poseStreamer.removePosesLessThan(std::min(meshWarpReceiver.poseIdColor, meshWarpReceiver.poseIdDepth));

        // Set render state
        node.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
        nodeWireframe.visible = renderState == RenderState::WIREFRAME;

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        tonemapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
