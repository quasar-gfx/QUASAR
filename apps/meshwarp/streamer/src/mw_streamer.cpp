#include <iostream>

#include <args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <VideoStreamer.h>
#include <DepthStreamer.h>
#include <PoseReceiver.h>

#define TEXTURE_PREVIEW_SIZE 500

enum class PauseState {
    PLAY,
    PAUSE_COLOR,
    PAUSE_DEPTH,
    PAUSE_BOTH
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Streamer";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "0.0.0.0:54321");
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
    args::ValueFlag<int> targetBitrateIn(parser, "targetBitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 50);
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
    config.showWindow = args::get(displayIn);

    std::string scenePath = args::get(scenePathIn);
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    unsigned int targetBitrate = args::get(targetBitrateIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene scene = Scene();
    PerspectiveCamera camera = PerspectiveCamera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, scene, camera);

    VideoStreamer videoStreamerColorRT = VideoStreamer({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL, targetBitrate);
    DepthStreamer videoStreamerDepthRT = DepthStreamer({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_R16,
        .format = GL_RED,
        .type = GL_UNSIGNED_SHORT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    }, depthURL);
    PoseReceiver poseReceiver = PoseReceiver(&camera, poseURL);

    PauseState pauseState = PauseState::PLAY;
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showDepth = true;

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);

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
            ImGui::MenuItem("Depth Preview", 0, &showDepth);
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
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Draw Calls: %d", renderStats.drawCalls);

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f fps), D (%.1f fps)", videoStreamerColorRT.getFrameRate(), videoStreamerDepthRT.getFrameRate());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: RGB (%.1f ms), D (%.1f ms)", videoStreamerColorRT.stats.timeToCopyFrameMs, videoStreamerDepthRT.stats.timeToCopyFrameMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: %.1f ms", videoStreamerColorRT.stats.timeToEncodeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: %.1f ms", videoStreamerColorRT.stats.timeToSendMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: RGB (%.1f Mbps), D (%.1f Mbps)", videoStreamerColorRT.stats.bitrateMbps, videoStreamerDepthRT.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::RadioButton("Play All", (int*)&pauseState, 0);
            ImGui::RadioButton("Pause Color", (int*)&pauseState, 1);
            ImGui::RadioButton("Pause Depth", (int*)&pauseState, 2);
            ImGui::RadioButton("Pause Both", (int*)&pauseState, 3);

            ImGui::End();
        }

        if (showDepth) {
            ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 40), ImGuiCond_FirstUseEver);
            flags = ImGuiWindowFlags_AlwaysAutoResize;
            ImGui::Begin("Raw Depth Texture", 0, flags);
            ImGui::Image((void*)(intptr_t)videoStreamerDepthRT.colorBuffer.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(winSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                if (saveAsHDR) {
                    renderer.gBuffer.saveColorAsHDR(fileName + ".hdr");
                }
                else {
                    renderer.gBuffer.saveColorAsPNG(fileName + ".png");
                }
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        renderer.resize(width, height);

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    // shaders
    Shader colorShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader depthShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "./shaders/displayDepth.frag"
    });

    // save camera view and projection matrices
    std::ofstream cameraFile;
    cameraFile.open("data/camera.bin", std::ios::out | std::ios::binary);
    glm::mat4 proj = camera.getProjectionMatrix();
    glm::mat4 view = camera.getViewMatrix();
    cameraFile.write(reinterpret_cast<const char*>(&proj), sizeof(glm::mat4));
    cameraFile.write(reinterpret_cast<const char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    // set fov
    camera.setFovy(glm::radians(args::get(fovIn)));

    // save remote camera view and projection matrices
    std::ofstream remoteCameraFile;
    remoteCameraFile.open("data/remoteCamera.bin", std::ios::out | std::ios::binary);
    proj = camera.getProjectionMatrix();
    view = camera.getViewMatrix();
    remoteCameraFile.write(reinterpret_cast<const char*>(&proj), sizeof(glm::mat4));
    remoteCameraFile.write(reinterpret_cast<const char*>(&view), sizeof(glm::mat4));
    remoteCameraFile.close();

    std::vector<glm::vec3> pointLightPositions(4);
    pointLightPositions[0] = scene.pointLights[0]->position;
    pointLightPositions[1] = scene.pointLights[1]->position;
    pointLightPositions[2] = scene.pointLights[2]->position;
    pointLightPositions[3] = scene.pointLights[3]->position;

    pose_id_t poseID = 0;
    app.onRender([&](double now, double dt) {
        if (pauseState == PauseState::PAUSE_BOTH) {
            return;
        }

        // receive pose
        poseID = poseReceiver.receivePose(false);

        // animate lights
        scene.pointLights[0]->setPosition(pointLightPositions[0] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        scene.pointLights[1]->setPosition(pointLightPositions[1] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        scene.pointLights[2]->setPosition(pointLightPositions[2] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));
        scene.pointLights[3]->setPosition(pointLightPositions[3] + glm::vec3(1.1f * sin(now), 0.0f, 0.0f));

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        if (config.showWindow) {
            renderer.drawToScreen(colorShader);
        }
        renderer.drawToRenderTarget(colorShader, videoStreamerColorRT);
        depthShader.bind();
        depthShader.setFloat("near", camera.near);
        depthShader.setFloat("far", camera.far);
        renderer.drawToRenderTarget(depthShader, videoStreamerDepthRT);

        // send video frame
        if (poseID != -1) {
            if (pauseState != PauseState::PAUSE_COLOR) videoStreamerColorRT.sendFrame(poseID);
            if (pauseState != PauseState::PAUSE_DEPTH) videoStreamerDepthRT.sendFrame(poseID);
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
