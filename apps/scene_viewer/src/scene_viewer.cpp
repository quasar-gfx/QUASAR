#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/DeferredRenderer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowDepthEffect.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <PostProcessing/ShowPositionsEffect.h>
#include <PostProcessing/ShowIDsEffect.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/ScreenshotWindow.h>
#include <UI/RecordWindow.h>
#include <UI/SceneWindow.h>
#include <UI/AnimationWindow.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Scene Viewer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> verbosity(parser, "verbosity", "Set log verbosity level", {'v', "verbosity"}, 2 /* spdlog::level::info */);
    args::ValueFlag<std::string> sizeIn(parser, "size", "Window resolution", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> rsizeIn(parser, "rsize", "Renderer resolution", {"rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<int> numPosesIn(parser, "num-poses", "Number of poses to load from camera path", {'n', "num-poses"}, -1);
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

    // Parse arguments
    bool saveImagesToDisk = args::get(saveImages);
    int numPoses = args::get(numPosesIn);
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);

    // Parse window size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    // Parse render size
    std::string rsizeStr = args::get(rsizeIn);
    pos = rsizeStr.find('x');
    glm::uvec2 renderSize = glm::uvec2(std::stoi(rsizeStr.substr(0, pos)), std::stoi(rsizeStr.substr(pos + 1)));

    config.verbosity = args::get(verbosity);
    config.enableVSync = !args::get(novsync) && !saveImagesToDisk;
    config.showWindow = !saveImagesToDisk;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    DeferredRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    // Post processing
    Tonemapper tonemapper;
    ShowDepthEffect showDepthEffect(camera, 10.0f);
    ShowNormalsEffect showNormalsEffect;
    ShowPositionsEffect showPositionsEffect;
    ShowIDsEffect showIDsEffect;

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
    }, renderer, tonemapper, outputPath, config.targetFramerate);

    float exposure = 1.0f;
    int shaderIndex = 0;
    double totalTime = 0.0;
    double totalDT = 0.0;

    FrameRateWindow frameRateWindow;
    ScreenshotWindow screenshotWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    SceneWindow sceneWindow(scene, ImVec2(430, 800));
    AnimationWindow animationWindow(ImVec2(430, 270));
    CameraHeader cameraHeader(camera);

    CameraAnimator cameraAnimator(cameraPathFile, numPoses, !saveImagesToDisk); // Disable tweening when saving images
    if (saveImagesToDisk || !cameraPathFile.str().empty()) {
        cameraAnimator.copyPoseToCamera(camera);
        animationWindow.setPlaying(true);
        spdlog::info("Loading camera path {} and saving images to {}", cameraPathFile.str(), outputPath.str());

        if (saveImagesToDisk) {
            recorder.setTargetFrameRate(-1 /* unlimited */);
            recorder.setFormat(Recorder::OutputFormat::PNG);
            recorder.start();
        }
    }

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImagesToDisk;

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
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Scene Settings", 0, &sceneWindow.visible);
            ImGui::MenuItem("Animations", 0, &animationWindow.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Frame Capture")) {
            ImGui::MenuItem("Take Screenshot", 0, &screenshotWindow.visible);
            ImGui::MenuItem("Record Video", 0, &recordWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        screenshotWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        sceneWindow.draw(now, dt);
        animationWindow.draw(now, dt);

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

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Post Processing Settings")) {
                ImGui::DragFloat("Exposure", &exposure, 0.1f, 0.1f, 5.0f);
                ImGui::RadioButton("Show Color", &shaderIndex, 0);
                ImGui::RadioButton("Show Depth", &shaderIndex, 1);
                ImGui::RadioButton("Show Normals", &shaderIndex, 2);
                ImGui::RadioButton("Show Positions", &shaderIndex, 3);
                ImGui::RadioButton("Show Object IDs", &shaderIndex, 4);
                ImGui::RadioButton("Show Primitive IDs", &shaderIndex, 5);
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

    double lastRenderTime = -INFINITY;
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

        if (cameraAnimator.isRunning()) {
            bool waypointUpdated = cameraAnimator.update(saveImagesToDisk ? (1.0 / 60.0) : dt);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (waypointUpdated) {
                cameraAnimator.copyPoseToCamera(camera);
            }
        }
        else {
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
            camera.processKeyboard(keys, dt);
        }
        if (animationWindow.isPlaying()) {
            totalTime += dt;
            totalDT += dt;
        }

        // Update all animations
        float animationInterval = animationWindow.getAnimationIntervalMs();
        if (animationInterval > 0.0 && (now - lastRenderTime) >= (animationInterval - 1.0) / MILLISECONDS_IN_SECOND) {
            if (animationWindow.isPlaying()) {
                scene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lastRenderTime = now;
        }

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        if (shaderIndex == 0) {
            tonemapper.setExposure(exposure);
            tonemapper.drawToScreen(renderer);
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

        if (saveImagesToDisk) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.isRunning()) {
                recorder.stop();
                window->close();
            }
        }
        else if (recordWindow.isRecording()) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
