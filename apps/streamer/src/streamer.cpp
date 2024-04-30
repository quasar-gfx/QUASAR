#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Primatives/Model.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <SceneLoader.h>

#include <VideoStreamer.h>
#include <PoseReceiver.h>

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Streamer";

    std::string outputUrl = "udp://127.0.0.1:1234";
    std::string poseURL = "udp://127.0.0.1:4321";
    std::string modelPath = "../assets/models/Sponza/Sponza.gltf";
    std::string hdrImagePath = "../assets/textures/hdr/barcelona.hdr";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            app.config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            app.config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            modelPath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            hdrImagePath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputUrl = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            poseURL = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            app.config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            app.config.showWindow = atoi(argv[i + 1]);
            i++;
        }
    }

    GLFWWindow window(app.config);
    app.init(&window);

    unsigned int screenWidth, screenHeight;
    window.getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene("../assets/scenes/sponza.json", scene, camera);

    VideoStreamer videoStreamer = VideoStreamer();
    PoseReceiver poseReceiver = PoseReceiver(&camera, poseURL);

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Text("Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamer.getFrameRate(), 1000.0f / videoStreamer.getFrameRate());
        ImGui::End();
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    // shaders
    Shader screenShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    int ret = videoStreamer.start(app.renderer.gBuffer.colorBuffer, outputUrl);
    if (ret < 0) {
        std::cerr << "Failed to initialize FFMpeg Video Streamer" << std::endl;
        return ret;
    }

    app.onRender([&](double now, double dt) {
        // handle mouse input
        auto mouseButtons = window.getMouseButtons();
        window.setMouseCursor(!mouseButtons.LEFT_PRESSED);
        static bool dragging = false;
        static bool prevMouseLeftPressed = false;
        static float lastX = screenWidth / 2.0;
        static float lastY = screenHeight / 2.0;
        if (!prevMouseLeftPressed && mouseButtons.LEFT_PRESSED) {
            dragging = true;
            prevMouseLeftPressed = true;

            auto cursorPos = window.getCursorPos();
            lastX = static_cast<float>(cursorPos.x);
            lastY = static_cast<float>(cursorPos.y);
        }
        if (prevMouseLeftPressed && !mouseButtons.LEFT_PRESSED) {
            dragging = false;
            prevMouseLeftPressed = false;
        }
        if (dragging) {
            auto cursorPos = window.getCursorPos();
            float xpos = static_cast<float>(cursorPos.x);
            float ypos = static_cast<float>(cursorPos.y);

            float xoffset = xpos - lastX;
            float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

            lastX = xpos;
            lastY = ypos;

            camera.processMouseMovement(xoffset, yoffset, true);
        }

        // handle keyboard input
        auto keys = window.getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window.close();
        }

        poseReceiver.receivePose();

        // animate lights
        scene.pointLights[0]->setPosition(glm::vec3(-1.45f + 1.1f * sin(now), 2.5f, -6.2f));
        scene.pointLights[1]->setPosition(glm::vec3(2.2f + 1.1f * sin(now), 2.5f, -6.2f));
        scene.pointLights[2]->setPosition(glm::vec3(-1.45f + 1.1f * sin(now), 2.5f, 4.89f));
        scene.pointLights[3]->setPosition(glm::vec3(2.2f + 1.1f * sin(now), 2.5f, 4.89f));

        // render all objects in scene
        app.renderer.drawObjects(scene, camera);

        // render to screen
        app.renderer.drawToScreen(screenShader);

        videoStreamer.sendFrame();
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
