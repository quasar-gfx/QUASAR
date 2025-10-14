#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowDepthEffect.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <PostProcessing/ShowPositionsEffect.h>
#include <PostProcessing/ShowIDsEffect.h>

#include <UI/FrameRateWindow.h>
#include <UI/FrameCaptureWindow.h>
#include <UI/RecordWindow.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Depth Peeling";

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

    if (verbose) spdlog::set_level(spdlog::level::debug);

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
    DepthPeelingRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    // Post processing
    Tonemapper tonemapper;
    ShowDepthEffect showDepthEffect(camera);
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
    CameraAnimator cameraAnimator(cameraPathFile);

    if (saveImages) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();

        cameraAnimator.copyPoseToCamera(camera);
    }

    float exposure = 1.0f;
    int shaderIndex = 0;

    bool runAnimations = cameraPathFileIn;
    float animationInterval = (MILLISECONDS_IN_SECOND / 30.0f);
    int animationFramerates[] = {1, 5, 10, 24, 30, 60};
    const char* animationFramerateLabels[] = {"1 FPS", "5 FPS", "10 FPS", "24 FPS", "30 FPS", "60 FPS"};
    int animationFramerateIndex = 4;

    double totalTime = 0.0;
    double totalDT = 0.0;

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, glm::uvec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, glm::uvec2(430, 270), outputPath);
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImages;
        static bool showLayerPreviews = false;
        static bool showAnimationsWindow = false;

        static bool showSkyBox = true;

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
            ImGui::MenuItem("Record", 0, &recordWindow.visible);
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Animations")) {
            ImGui::MenuItem("Animations", 0, &showAnimationsWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);

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

            const glm::vec3& position = camera.getPosition();
            if (ImGui::DragFloat3("Camera Position", (float*)&position, 0.01f)) {
                camera.setPosition(position);
            }
            const glm::vec3& rotation = camera.getRotationEuler();
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
                ImGui::ColorEdit3("Color##Ambient", (float*)&scene.ambientLight->color);
                ImGui::DragFloat("Strength##Ambient", &scene.ambientLight->intensity, 0.05f, 0.1f, 2.0f);
            }

            if (scene.directionalLight != nullptr && ImGui::CollapsingHeader("Directional Light Settings")) {
                ImGui::ColorEdit3("Color##Directional", (float*)&scene.directionalLight->color);
                ImGui::DragFloat("Strength##Directional", &scene.directionalLight->intensity, 0.1f, 0.1f, 100.0f);
                ImGui::DragFloat3("Direction##Directional", (float*)&scene.directionalLight->direction, 0.1f, -5.0f, 5.0f);
                ImGui::DragFloat("Distance##Directional", &scene.directionalLight->distance, 0.1f, 0.0f, 100.0f);

                ImGui::TextColored(ImVec4(1,1,1,1), "Shadow Map:");
                int halfWindowWidth = ImGui::GetWindowWidth() / 2;
                ImGui::Image(
                    (void*)(intptr_t)scene.directionalLight->shadowMapRenderTarget.depthTexture,
                    ImVec2(halfWindowWidth, halfWindowWidth), ImVec2(0, 1), ImVec2(1, 0));
            }

            if (scene.pointLights.size() > 0 && ImGui::CollapsingHeader("Point Lights")) {
                for (size_t i = 0; i < scene.pointLights.size(); i++) {
                    PointLight* pointLight = scene.pointLights[i];
                    if (ImGui::TreeNode((void*)(intptr_t)i, "Point Light %ld", i)) {
                        ImGui::ColorEdit3(("Color##Point" + std::to_string(i)).c_str(), (float*)&pointLight->color);
                        ImGui::DragFloat(("Strength##Point" + std::to_string(i)).c_str(), &pointLight->intensity, 0.1f, 0.1f, 100.0f);
                        ImGui::DragFloat3(("Position##Point" + std::to_string(i)).c_str(), (float*)&pointLight->position, 0.01f);
                        ImGui::DragFloat(("Constant##Point" + std::to_string(i)).c_str(), &pointLight->constant, 0.01f, 0.0f, 10.0f);
                        ImGui::DragFloat(("Linear##Point" + std::to_string(i)).c_str(), &pointLight->linear, 0.001f, 0.0f, 1.0f);
                        ImGui::DragFloat(("Quadratic##Point" + std::to_string(i)).c_str(), &pointLight->quadratic, 0.001f, 0.0f, 1.0f);
                        ImGui::Checkbox(("Debug##Point" + std::to_string(i)).c_str(), &pointLight->debug);
                        ImGui::TreePop();
                    }
                }
            }

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

        frameCaptureWindow.draw(now, dt);
        recordWindow.draw(now, dt);

        if (showLayerPreviews) {
            for (int layer = 0; layer < renderer.maxLayers; layer++) {
                int layerIdx = renderer.maxLayers - layer - 1;
                ImGui::Begin(("Layer " + std::to_string(layerIdx)).c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::Image((void*)(intptr_t)(renderer.peelingLayers[layerIdx].colorTexture.ID),
                                ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                ImGui::End();
            }
        }

        if (showAnimationsWindow) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Animations", &showAnimationsWindow);

            ImGui::TextColored(ImVec4(0,1,0,1), "Current Time: %.3f s", totalTime);

            ImGui::Separator();

            ImGui::Text("Animation Framerate:");
            int animationFramerate = animationFramerates[animationFramerateIndex];
            if (ImGui::Combo("", &animationFramerateIndex, animationFramerateLabels, IM_ARRAYSIZE(animationFramerateLabels))) {
                animationFramerate = animationFramerates[animationFramerateIndex];
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
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });

    double lastRenderTime = -INFINITY;
    bool updateClient = !saveImages;
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
        else if (recordWindow.isRecording()) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
