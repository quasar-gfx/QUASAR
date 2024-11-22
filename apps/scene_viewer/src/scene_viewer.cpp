#include <iostream>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Recorder.h>
#include <Animator.h>

#include <Shaders/ToneMapShader.h>

#include <Utils/Utils.h>

const std::string DATA_PATH = "./";

int main(int argc, char** argv) {
    Config config{};
    config.title = "Scene Viewer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<std::string> pathFileIn(parser, "path", "Path to camera animation file", {'p', "path"});
    args::ValueFlag<int> frameRateIn(parser, "frame-rate", "Frame rate", {'f', "frame-rate"}, 30);

    args::Flag saveImage(parser, "save", "Save image and exit", {'b', "save-image"});
    args::PositionalList<float> poseOffset(parser, "pose-offset", "Offset for the pose (only used when --save-image is set)");
    char recordingPath[256] = "./recordings";
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

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);
    config.showWindow = !args::get(saveImage);

    std::string scenePath = args::get(scenePathIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    // Define the output path for the recorded frames
    std::string outputPath = "./recordings";

    // Create the Recorder instance
    Recorder recorder(args::get(frameRateIn), outputPath, renderer); // Capturing every 1 second


    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    SceneLoader loader;
    loader.loadScene(scenePath, scene, camera);


    std::shared_ptr<Animator> animator;

    if (pathFileIn) {
        std::string pathFile = args::get(pathFileIn);
        animator = std::make_shared<Animator>(pathFile);
    }
    // shaders
    ToneMapShader toneMapShader;

    Shader showDepthShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYDEPTH_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYDEPTH_FRAG_len
    });

    Shader showPositionShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYPOSITIONS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYPOSITIONS_FRAG_len
    });

    Shader showNormalShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYNORMALS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYNORMALS_FRAG_len
    });

    Shader showIDShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYIDS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYIDS_FRAG_len
    });

    float exposure = 1.0f;
    int shaderIndex = 0;
    RenderStats renderStats;
    bool recording = false;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool showRecordWindow = false;
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

            if (scene.directionalLight != nullptr && ImGui::CollapsingHeader("Directional Light Settings")) {
                ImGui::TextColored(ImVec4(1,1,1,1), "Directional Light Settings");
                ImGui::ColorEdit3("Color", (float*)&scene.directionalLight->color);
                ImGui::SliderFloat("Strength", &scene.directionalLight->intensity, 0.1f, 100.0f);
                ImGui::SliderFloat3("Direction", (float*)&scene.directionalLight->direction, -5.0f, 5.0f);
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Post Processing Settings")) {
                ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f);
                ImGui::RadioButton("Show Color", &shaderIndex, 0);
                ImGui::RadioButton("Show Depth", &shaderIndex, 1);
                ImGui::RadioButton("Show Positions", &shaderIndex, 2);
                ImGui::RadioButton("Show Normals", &shaderIndex, 3);
                ImGui::RadioButton("Show Primative IDs", &shaderIndex, 4);
            }

            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = DATA_PATH + std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                saveRenderTargetToFile(renderer, toneMapShader, fileName, windowSize, saveAsHDR);
            }

            ImGui::End();
        }

        if (showRecordWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Record", &showRecordWindow);
            
            ImGui::Text("Output Directory:");
            ImGui::InputText("##output directory", recordingPath, IM_ARRAYSIZE(recordingPath));
            
            static int fps = 30;  // Default FPS
            ImGui::Text("FPS:");
            if (ImGui::InputInt("##fps", &fps)) {
                fps = std::max(1, fps);  // Ensure FPS is at least 1
                recorder.setFrameRate(fps);
            }

            static int formatIndex = 0;  // 0: PNG, 1: JPG, 2: MP4
            const char* formats[] = { "PNG", "JPG", "MP4" };
            ImGui::Text("Save Format:");
            if (ImGui::Combo("##format", &formatIndex, formats, IM_ARRAYSIZE(formats))) {
                OutputFormat selectedFormat = OutputFormat::PNG;
                switch (formatIndex) {
                    case 0: selectedFormat = OutputFormat::PNG; break;
                    case 1: selectedFormat = OutputFormat::JPG; break;
                    case 2: selectedFormat = OutputFormat::MP4; break;
                }
                recorder.setOutputFormat(selectedFormat);
            }
            
            ImGui::Separator();
            ImGui::Text("Record Frames");
            ImGui::Separator();
            
            if (ImGui::Button("Record")) {
                recording = true;
                recorder.setOutputPath(recordingPath);
                recorder.start();
            }
            if (ImGui::Button("Stop")) {
                recording = false;
                recorder.stop();
            }
            ImGui::End();
        }
        if (recording) {
            recorder.captureFrame(renderer.gBuffer, camera);
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.resize(windowSize.x, windowSize.y);

        camera.aspect = (float)windowSize.x / (float)windowSize.y;
        camera.updateProjectionMatrix();
    });

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
        if (animator && !animator->isFinished()) {
            animator->update(dt);
            glm::vec3 position = animator->getCurrentPosition();

            glm::quat rotation = animator->getCurrentRotation();
            camera.setPosition(position);
            camera.setRotationQuat(rotation);
            camera.updateViewMatrix();
        } else {
            // handle keyboard input
            camera.processKeyboard(keys, dt);
        }
        if (animator && !animator->isFinished()) {
            animator->update(dt);
            glm::vec3 position = animator->getCurrentPosition();

            glm::quat rotation = animator->getCurrentRotation();
            camera.setPosition(position);
            camera.setRotationQuat(rotation);
            camera.updateViewMatrix();
        } else {
            // handle keyboard input
            camera.processKeyboard(keys, dt);
        }

        if (saveImage && args::get(poseOffset).size() == 6) {
            glm::vec3 positionOffset, rotationOffset;
            for (int i = 0; i < 3; i++) {
                positionOffset[i] = args::get(poseOffset)[i];
                rotationOffset[i] = args::get(poseOffset)[i + 3];
            }
            camera.setPosition(camera.getPosition() + positionOffset);
            camera.setRotationEuler(camera.getRotationEuler() + rotationOffset);
            camera.updateViewMatrix();
        }

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);


        // render to screen
        if (shaderIndex == 1) {
            showDepthShader.bind();
            showDepthShader.setFloat("near", camera.near);
            showDepthShader.setFloat("far", camera.far);
            renderer.drawToScreen(showDepthShader);
        }
        else if (shaderIndex == 2) {
            showPositionShader.bind();
            renderer.drawToScreen(showPositionShader);
        }
        else if (shaderIndex == 3) {
            showNormalShader.bind();
            renderer.drawToScreen(showNormalShader);
        }
        else if (shaderIndex == 4) {
            showIDShader.bind();
            renderer.drawToScreen(showIDShader);
        }
        else {
            toneMapShader.bind();
            toneMapShader.setFloat("exposure", exposure);
            renderer.drawToScreen(toneMapShader);

            if (saveImage) {
                glm::vec3 position = camera.getPosition();
                glm::vec3 rotation = camera.getRotationEuler();
                std::string positionStr = to_string_with_precision(position.x) + "_" + to_string_with_precision(position.y) + "_" + to_string_with_precision(position.z);
                std::string rotationStr = to_string_with_precision(rotation.x) + "_" + to_string_with_precision(rotation.y) + "_" + to_string_with_precision(rotation.z);

                std::cout << "Saving output with pose: Position(" << positionStr << ") Rotation(" << rotationStr << ")" << std::endl;

                std::string fileName = DATA_PATH + "screenshot." + positionStr + "_" + rotationStr;
                saveRenderTargetToFile(renderer, toneMapShader, fileName, windowSize);
                window->close();
            }
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}