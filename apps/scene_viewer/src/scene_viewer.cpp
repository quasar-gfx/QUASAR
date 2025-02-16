#include <filesystem>

#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowDepthEffect.h>
#include <PostProcessing/ShowPositionsEffect.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <PostProcessing/ShowIDsEffect.h>

#include <Recorder.h>
#include <Animator.h>

int main(int argc, char** argv) {
    Config config{};
    config.title = "Scene Viewer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'V', "vsync"}, true);
    args::Flag saveImage(parser, "save", "Save outputs to disk", {'I', "save-image"});
    args::ValueFlag<std::string> animationFileIn(parser, "anim-path", "Path to camera animation file", {'A', "animation-path"});
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Directory to save data", {'D', "data-path"}, ".");
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
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = args::get(vsyncIn);
    config.showWindow = !args::get(saveImage);

    std::string sceneFile = args::get(sceneFileIn);
    std::string animationFile = args::get(animationFileIn);
    std::string dataPath = args::get(dataPathIn);
    if (dataPath.back() != '/') {
        dataPath += "/";
    }
    // create data path if it doesn't exist
    if (!std::filesystem::exists(dataPath)) {
        std::filesystem::create_directories(dataPath);
    }

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, scene, camera);

    // post processing
    ToneMapper toneMapper;
    ShowDepthEffect ShowDepthEffect(camera);
    ShowPositionsEffect showPositionsEffect;
    ShowNormalsEffect showNormalsEffect;
    ShowIDsEffect showIDsEffect;

    Recorder recorder(renderer, toneMapper, dataPath, config.targetFramerate);
    Animator animator(animationFile);

    if (saveImage) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();

        animator.copyPoseToCamera(camera);
    }

    float exposure = 1.0f;
    int shaderIndex = 0;
    bool recording = false;
    float animationInterval = animationFileIn ? (MILLISECONDS_IN_SECOND / 30.0) : -1.0; // run at 30fps if animation file is provided
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool showRecordWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static int recordingFormatIndex = 0;
        static const char* formats[] = { "MP4", "PNG", "JPG" };
        static char recordingDirBase[256] = "recordings";

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
            ImGui::MenuItem("Record", 0, &showRecordWindow);
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
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 0.1f, 20.0f);

            ImGui::Separator();

            if (scene.ambientLight != nullptr && ImGui::CollapsingHeader("Ambient Light Settings")) {
                ImGui::TextColored(ImVec4(1,1,1,1), "Ambient Light Settings");
                ImGui::ColorEdit3("Color", (float*)&scene.ambientLight->color);
                ImGui::SliderFloat("Strength", &scene.ambientLight->intensity, 0.1f, 1.0f);
            }

            if (scene.directionalLight != nullptr && ImGui::CollapsingHeader("Directional Light Settings")) {
                ImGui::TextColored(ImVec4(1,1,1,1), "Directional Light Settings");
                ImGui::ColorEdit3("Color", (float*)&scene.directionalLight->color);
                ImGui::SliderFloat("Strength", &scene.directionalLight->intensity, 0.1f, 100.0f);
                ImGui::SliderFloat3("Direction", (float*)&scene.directionalLight->direction, -5.0f, 5.0f);
                ImGui::SliderFloat("Distance", &scene.directionalLight->distance, 0.0f, 100.0f);
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Post Processing Settings")) {
                ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f);
                ImGui::RadioButton("Show Color", &shaderIndex, 0);
                ImGui::RadioButton("Show Depth", &shaderIndex, 1);
                ImGui::RadioButton("Show Positions", &shaderIndex, 2);
                ImGui::RadioButton("Show Normals", &shaderIndex, 3);
                ImGui::RadioButton("Show Object IDs", &shaderIndex, 4);
                ImGui::RadioButton("Show Primative IDs", &shaderIndex, 5);
            }

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = dataPath + std::string(fileNameBase) + "." +
                                              std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);
            }

            ImGui::End();
        }

        if (showRecordWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Record", &showRecordWindow);

            if (recording) {
                ImGui::TextColored(ImVec4(1,0,0,1), "Recording in progress...");
            }

            ImGui::Text("Output Directory:");
            ImGui::InputText("##output directory", recordingDirBase, IM_ARRAYSIZE(recordingDirBase));

            ImGui::Text("FPS:");
            if (ImGui::InputInt("##fps", &recorder.targetFrameRate)) {
                recorder.setTargetFrameRate(recorder.targetFrameRate);
            }

            ImGui::Text("Format:");
            if (ImGui::Combo("##format", &recordingFormatIndex, formats, IM_ARRAYSIZE(formats))) {
                Recorder::OutputFormat selectedFormat = Recorder::OutputFormat::MP4;
                switch (recordingFormatIndex) {
                    case 0: selectedFormat = Recorder::OutputFormat::MP4; break;
                    case 1: selectedFormat = Recorder::OutputFormat::PNG; break;
                    case 2: selectedFormat = Recorder::OutputFormat::JPG; break;
                    default: break;
                }
                recorder.setFormat(selectedFormat);
            }

            if (ImGui::Button("Start")) {
                recording = true;
                std::string recordingDir = dataPath + std::string(recordingDirBase) + "." +
                                                      std::to_string(static_cast<int>(window->getTime() * 1000.0f));
                recorder.setOutputPath(recordingDir);
                recorder.start();
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop")) {
                recorder.stop();
                recording = false;
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

    double lastRenderTime = 0.0;
    double totalDT = 0.0;
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
        auto scroll = window->getScrollOffset();
        camera.processScroll(scroll.y);

        if (animator.running) {
            animator.copyPoseToCamera(camera);
            animator.update(dt);
        }
        else {
            // handle keyboard input
            camera.processKeyboard(keys, dt);
        }

        if (animationFileIn) {
            now = animator.now;
            dt = animator.dt;
        }

        totalDT += dt;

        // update all animations
        if (animationInterval == -1.0 || now - lastRenderTime >= animationInterval / MILLISECONDS_IN_SECOND) {
            scene.updateAnimations(totalDT);
            lastRenderTime = now;
            totalDT = 0.0;
        }

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        if (shaderIndex == 0) {
            toneMapper.setExposure(exposure);
            toneMapper.drawToScreen(renderer);
        }
        else if (shaderIndex == 1) {
            ShowDepthEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 2) {
            showPositionsEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 3) {
            showNormalsEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 4) {
            showIDsEffect.showObjectIDs(true);
            showIDsEffect.drawToScreen(renderer);
        }
        else if (shaderIndex == 5) {
            showIDsEffect.showObjectIDs(false);
            showIDsEffect.drawToScreen(renderer);
        }

        if ((animationFileIn && animator.running) || recording) {
            recorder.captureFrame(camera);
        }
        if (animationFileIn && !animator.running) {
            recorder.captureFrame(camera); // capture final frame
            recorder.stop();
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
