
#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/DeferredRenderer.h>

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowDepthEffect.h>

#include <CameraAnimator.h>

#include <Shaders/ComputeShader.h>
#include <VideoStreamer.h>
#include <DepthStreamer.h>
#include <BC4DepthStreamer.h>
#include <PoseReceiver.h>

using namespace quasar;

enum class PauseState {
    PLAY,
    PAUSE_COLOR,
    PAUSE_DEPTH,
    PAUSE_BOTH
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Streamer";
    config.targetFramerate = 30;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "0.0.0.0:54321");
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
    args::ValueFlag<int> targetBitrateIn(parser, "targetBitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 50);
    args::ValueFlag<int> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
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

    config.enableVSync = !args::get(novsync);
    config.showWindow = args::get(displayIn);

    std::string sceneFile = args::get(sceneFileIn);
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    uint targetBitrate = args::get(targetBitrateIn);
    int depthFactor = args::get(depthFactorIn);

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

    // Set fov
    camera.setFovyDegrees(args::get(fovIn));

    glm::vec3 initialPosition = camera.getPosition();

    VideoStreamer videoStreamerColorRT = VideoStreamer({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoURL, config.targetFramerate, targetBitrate);

    BC4DepthStreamer bc4DepthStreamerRT = BC4DepthStreamer({
        .width = windowSize.x / depthFactor,
        .height = windowSize.y / depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthURL);

    PoseReceiver poseReceiver = PoseReceiver(&camera, poseURL);

    // Post processing
    ToneMapper toneMapper;
    ShowDepthEffect showDepthEffect(camera);

    PauseState pauseState = PauseState::PLAY;
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;

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
            glm::vec3 rotation = camera.getRotationEuler();
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Camera Position", (float*)&position);
            ImGui::DragFloat3("Camera Rotation", (float*)&rotation);
            ImGui::EndDisabled();

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Depth URL: %s", depthURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f fps), BC4 D (%.1f fps)", videoStreamerColorRT.getFrameRate(), bc4DepthStreamerRT.getFrameRate());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: RGB (%.3f ms), BC4 D (%.3f ms)", videoStreamerColorRT.stats.timeToCopyFrameMs, bc4DepthStreamerRT.stats.timeToCopyFrameMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: RGB (%.3f ms), BC4 D (%.3f ms)", videoStreamerColorRT.stats.timeToEncodeMs, bc4DepthStreamerRT.stats.timeToCompressMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: RGB (%.3f ms), BC4 D (%.3f ms)", videoStreamerColorRT.stats.timeToSendMs, bc4DepthStreamerRT.stats.timeToSendMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: RGB (%.3f Mbps), BC4 D (%.3f Mbps)", videoStreamerColorRT.stats.bitrateMbps, bc4DepthStreamerRT.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::RadioButton("Play All", (int*)&pauseState, 0);
            ImGui::RadioButton("Pause Color", (int*)&pauseState, 1);
            ImGui::RadioButton("Pause Depth", (int*)&pauseState, 2);
            ImGui::RadioButton("Pause Both", (int*)&pauseState, 3);

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    app.onRender([&](double now, double dt) {
        // Handle keyboard input
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (pauseState == PauseState::PAUSE_BOTH) {
            return;
        }

        // Update all animations
        scene.updateAnimations(dt);

        // Receive pose
        pose_id_t poseID = poseReceiver.receivePose(false);
        if (poseID != -1) {

            // Offset camera
            camera.setPosition(camera.getPosition() + initialPosition);
            camera.updateViewMatrix();

            // Render all objects in scene
            renderStats = renderer.drawObjects(scene, camera);

            // Restore camera position
            camera.setPosition(camera.getPosition() - initialPosition);
            camera.updateViewMatrix();

            // Copy color and depth to video frames
            toneMapper.drawToRenderTarget(renderer, videoStreamerColorRT);
            showDepthEffect.drawToRenderTarget(renderer, bc4DepthStreamerRT);

            if (pauseState != PauseState::PAUSE_COLOR) videoStreamerColorRT.sendFrame(poseID);
            if (pauseState != PauseState::PAUSE_DEPTH) bc4DepthStreamerRT.sendFrame(poseID);
        }

        // Render to screen
        if (config.showWindow) {
            toneMapper.drawToScreen(renderer);
        }
    });

    // Run app loop (blocking)
    app.run();

    spdlog::info("Please do CTRL-C to exit!");

    return 0;
}
