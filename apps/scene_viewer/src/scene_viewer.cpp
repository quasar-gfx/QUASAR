#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/DeferredRenderer.h>

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowDepthEffect.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <PostProcessing/ShowPositionsEffect.h>
#include <PostProcessing/ShowIDsEffect.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Scene Viewer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
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

    config.enableVSync = !args::get(novsync) && !saveImages;
    config.showWindow = !args::get(saveImages);

    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    DeferredRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    // Post processing
    ToneMapper toneMapper;
    ShowDepthEffect showDepthEffect(camera);
    ShowNormalsEffect showNormalsEffect;
    ShowPositionsEffect showPositionsEffect;
    ShowIDsEffect showIDsEffect;

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
    }, renderer, toneMapper, outputPath, config.targetFramerate);
    CameraAnimator cameraAnimator(cameraPathFile);

    if (saveImages) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();

        cameraAnimator.copyPoseToCamera(camera);
    }

    float exposure = 1.0f;
    int shaderIndex = 0;
    bool recording = false;
    bool runAnimations = cameraPathFileIn;
    float animationInterval = (MILLISECONDS_IN_SECOND / 30.0f); // run animations at 30 FPS

    double totalTime = 0.0;
    double totalDT = 0.0;
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = !saveImages;
        static bool showFrameCaptureWindow = false;
        static char fileNameBase[256] = "screenshot";
        static bool saveAsHDR = false;
        static bool showRecordWindow = false;
        static int recordingFormatIndex = 0;
        static char recordingDirBase[256] = "recordings";
        static bool showAnimationsWindow = false;

        static int allowedFramerates[] = {1, 5, 10, 24, 30, 60};
        static const char* framerateLabels[] = {"1 FPS", "5 FPS", "10 FPS", "24 FPS", "30 FPS", "60 FPS"};
        static int currentFramerateIndex = 4;

        static bool showSkyBox = true;

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
            ImGui::MenuItem("Record", 0, &showRecordWindow);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Animations")) {
            ImGui::MenuItem("Animations", 0, &showAnimationsWindow);
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
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %d", renderStats.drawCalls);

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
                if (ImGui::Checkbox("Show Sky Box", &showSkyBox)) {
                    scene.envCubeMap = showSkyBox ? scene.envCubeMap : nullptr;
                }

                if (ImGui::Button("Change Background Color", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    ImGui::OpenPopup("Background Color Popup");
                }
                if (ImGui::BeginPopup("Background Color Popup")) {
                    ImGui::ColorPicker3("Background Color", (float*)&scene.backgroundColor);
                    ImGui::EndPopup();
                }
            }

            ImGui::Separator();

            if (scene.ambientLight != nullptr && ImGui::CollapsingHeader("Ambient Light Settings")) {
                ImGui::ColorEdit3("Color", (float*)&scene.ambientLight->color);
                ImGui::DragFloat("Strength", &scene.ambientLight->intensity, 0.1f, 0.1f, 1.0f);
            }

            if (scene.directionalLight != nullptr && ImGui::CollapsingHeader("Directional Light Settings")) {
                ImGui::ColorEdit3("Color", (float*)&scene.directionalLight->color);
                ImGui::DragFloat("Strength", &scene.directionalLight->intensity, 0.1f, 0.1f, 100.0f);
                ImGui::DragFloat3("Direction", (float*)&scene.directionalLight->direction, 0.1f, -5.0f, 5.0f);
                ImGui::DragFloat("Distance", &scene.directionalLight->distance, 0.1f, 0.0f, 100.0f);

                ImGui::TextColored(ImVec4(1,1,1,1), "Shadow Map:");
                int halfWindowWidth = ImGui::GetWindowWidth() / 2;
                ImGui::Image(
                    (void*)(intptr_t)scene.directionalLight->shadowMapRenderTarget.depthBuffer,
                    ImVec2(halfWindowWidth, halfWindowWidth), ImVec2(0, 1), ImVec2(1, 0));
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Post Processing Settings")) {
                ImGui::DragFloat("Exposure", &exposure, 0.1f, 0.1f, 5.0f);
                ImGui::RadioButton("Show Color", &shaderIndex, 0);
                ImGui::RadioButton("Show Depth", &shaderIndex, 1);
                ImGui::RadioButton("Show Normals", &shaderIndex, 2);
                ImGui::RadioButton("Show Positions", &shaderIndex, 3);
                ImGui::RadioButton("Show Object IDs", &shaderIndex, 4);
                ImGui::RadioButton("Show Primative IDs", &shaderIndex, 5);
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
            Path filename = (outputPath / fileNameBase).appendToName("." + time);

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(filename, saveAsHDR);
            }

            ImGui::End();
        }

        if (showRecordWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Record", &showRecordWindow);

            if (recording) { ImGui::TextColored(ImVec4(1,0,0,1), "Recording in progress..."); }
            else { ImGui::TextColored(ImVec4(1,0.5f,0,1), "No recording in progress"); }

            ImGui::Text("Output Directory:");
            ImGui::InputText("##output directory", recordingDirBase, IM_ARRAYSIZE(recordingDirBase));

            ImGui::Text("FPS:");
            if (ImGui::InputInt("##fps", &recorder.targetFrameRate)) {
                recorder.setTargetFrameRate(recorder.targetFrameRate);
            }

            ImGui::Text("Format:");
            if (ImGui::Combo("##format", &recordingFormatIndex, recorder.getFormatCStrArray(), recorder.getFormatCount())) {
                Recorder::OutputFormat selectedFormat = Recorder::OutputFormat::MP4;
                switch (recordingFormatIndex) {
                    case 0: selectedFormat = Recorder::OutputFormat::MP4; break;
                    case 1: selectedFormat = Recorder::OutputFormat::PNG; break;
                    case 2: selectedFormat = Recorder::OutputFormat::JPG; break;
                    default: break;
                }
                recorder.setFormat(selectedFormat);
            }

            if (!recording) {
                if (ImGui::Button("Start", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
                    Path recordingDir = (outputPath / recordingDirBase).appendToName("." + time);
                    spdlog::info("Recording to: {}", recordingDir.str());

                    recorder.setOutputPath(recordingDir);
                    recorder.start();

                    recording = true;
                }
            }
            else {
                if (ImGui::Button("Stop", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                    recorder.stop();
                    recording = false;
                }
            }

            ImGui::End();
        }

        if (showAnimationsWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Animations", &showAnimationsWindow);

            ImGui::TextColored(ImVec4(0,1,0,1), "Current Time: %.3f s", totalTime);

            ImGui::Separator();

            ImGui::Text("Animation Framerate:");
            int animationFramerate = allowedFramerates[currentFramerateIndex];
            if (ImGui::Combo("", &currentFramerateIndex, framerateLabels, IM_ARRAYSIZE(framerateLabels))) {
                animationFramerate = allowedFramerates[currentFramerateIndex];
                animationInterval = MILLISECONDS_IN_SECOND / static_cast<float>(animationFramerate);
            }

            ImGui::Separator();

            if (!runAnimations) {
                if (ImGui::Button("Play", ImVec2(ImGui::GetContentRegionAvail().x, 0))) { runAnimations = true; }
            }
            else {
                if (ImGui::Button("Pause", ImVec2(ImGui::GetContentRegionAvail().x, 0))) { runAnimations = false; }
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

    double lastRenderTime = -INFINITY;
    bool updateClient = !saveImages;
    app.onRender([&](double now, double dt) {
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
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (cameraAnimator.running) {
            updateClient = cameraAnimator.update(!cameraPathFileIn ? dt : 1.0 / MILLISECONDS_IN_SECOND);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (updateClient) {
                cameraAnimator.copyPoseToCamera(camera);
            }
        }
        else {
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
            camera.processKeyboard(keys, dt);
        }
        if (runAnimations) {
            totalTime += dt;
            totalDT += dt;
        }

        // Update all animations
        if (animationInterval > 0.0 && (now - lastRenderTime) >= (animationInterval - 1.0) / MILLISECONDS_IN_SECOND) {
            if (runAnimations) {
                scene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lastRenderTime = now;
        }

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        if (shaderIndex == 0) {
            toneMapper.setExposure(exposure);
            toneMapper.drawToScreen(renderer);
        }
        else if (shaderIndex == 1) {
            showDepthEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 2) {
            showNormalsEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 3) {
            showPositionsEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 4) {
            showIDsEffect.showObjectIDs(true);
            showIDsEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 5) {
            showIDsEffect.showObjectIDs(false);
            showIDsEffect.drawToScreen(renderer);
        }
        if (!updateClient) {
            return;
        }

        if (cameraPathFileIn) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.running) {
                recorder.stop();
                window->close();
            }
        }
        else if (recording) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
