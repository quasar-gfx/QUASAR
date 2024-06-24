#include <iostream>

#include <args.hxx>
#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <RenderTargets/RenderTarget.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <SceneLoader.h>

#include <VideoStreamer.h>
#include <DepthSender.h>
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
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'i', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "0.0.0.0:54321");
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
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
    size_t pos = sizeStr.find("x");
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);
    config.showWindow = args::get(displayIn);

    std::string scenePath = args::get(scenePathIn);
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, scene, camera);

    VideoStreamer videoStreamerColorRT = VideoStreamer({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_SRGB,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL);
    DepthSender videoStreamerDepthRT = DepthSender({
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

    std::cout << "Video URL: " << videoURL << std::endl;
    std::cout << "Depth URL: " << depthURL << std::endl;
    std::cout << "Pose URL: " << poseURL << std::endl;

    PauseState pauseState = PauseState::PLAY;
    guiManager->onRender([&](double now, double dt) {
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        int flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("", 0, flags);
        ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);
        glm::vec2 guiSize = winSize * glm::vec2(0.4f, 0.3f);
        ImGui::SetNextWindowSize(ImVec2(guiSize.x, guiSize.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 60), ImGuiCond_FirstUseEver);
        flags = 0;
        ImGui::Begin(config.title.c_str(), 0, flags);
        ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

        ImGui::Separator();

        ImGui::Text("Video URL: %s", videoURL.c_str());
        ImGui::Text("Pose URL: %s", poseURL.c_str());

        ImGui::Separator();

        ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamerColorRT.getFrameRate(), 1000.0f / videoStreamerColorRT.getFrameRate());

        ImGui::Separator();

        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: %.3f ms", videoStreamerColorRT.stats.timeToCopyFrameMs + videoStreamerDepthRT.stats.timeToCopyFrameMs);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: %.3f ms", videoStreamerColorRT.stats.timeToEncodeMs);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: %.3f ms", videoStreamerColorRT.stats.timeToSendMs);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: %.3f Mbps", videoStreamerColorRT.stats.bitrateMbps + videoStreamerDepthRT.stats.bitrateMbps);

        ImGui::Separator();

        ImGui::RadioButton("Play All", (int*)&pauseState, 0);
        ImGui::RadioButton("Pause Color", (int*)&pauseState, 1);
        ImGui::RadioButton("Pause Depth", (int*)&pauseState, 2);
        ImGui::RadioButton("Pause Both", (int*)&pauseState, 3);

        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 10), ImGuiCond_FirstUseEver);
        flags = ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::Begin("Raw Depth Texture", 0, flags);
        ImGui::Image((void*)(intptr_t)videoStreamerDepthRT.colorBuffer.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();
    });

    // shaders
    Shader colorShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader depthShader = Shader({
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
        // handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;
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

            // handle keyboard input
            auto keys = window->getKeys();
            camera.processKeyboard(keys, dt);
            if (keys.ESC_PRESSED) {
                window->close();
            }
        }

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
        app.renderer->drawObjects(scene, camera);

        // render to screen
        if (config.showWindow) {
            app.renderer->drawToScreen(colorShader);
        }
        app.renderer->drawToRenderTarget(colorShader, videoStreamerColorRT);
        depthShader.bind();
        depthShader.setFloat("near", camera.near);
        depthShader.setFloat("far", camera.far);
        app.renderer->drawToRenderTarget(depthShader, videoStreamerDepthRT);

        // send video frame
        if (poseID != -1) {
            if (pauseState != PauseState::PAUSE_COLOR) videoStreamerColorRT.sendFrame(poseID);
            if (pauseState != PauseState::PAUSE_DEPTH) videoStreamerDepthRT.sendFrame(poseID);
        }
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
