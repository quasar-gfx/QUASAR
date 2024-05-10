#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Materials/DiffSpecMaterial.h>
#include <CubeMap.h>
#include <Primatives/Entity.h>
#include <Scene.h>
#include <Camera.h>
#include <Framebuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>

#include <VideoTexture.h>
#include <PoseStreamer.h>

const std::string CONTAINER_TEXTURE = "../assets/textures/container.jpg";
const std::string METAL_TEXTURE = "../assets/textures/metal.png";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Receiver";
    app.config.sRGB = false;

    std::string inputUrl = "udp://127.0.0.1:1234";
    std::string poseURL = "udp://127.0.0.1:4321";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            app.config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            app.config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            inputUrl = argv[i + 1];
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
    }

    GLFWWindow window(app.config);
    app.init(&window);

    unsigned int screenWidth, screenHeight;
    window.getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

    TextureCreateParams videoParams{
        .width = app.config.width,
        .height = app.config.height,
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    };
    VideoTexture videoTexture(videoParams);
    videoTexture.initVideo(inputUrl);
    PoseStreamer poseStreamer(&camera, poseURL);

    ImGui::GetIO().Fonts->AddFontFromFileTTF("../assets/fonts/trebucbd.ttf", 24.0f);
    app.gui([&](double now, double dt) {
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        int flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("", 0, flags);
        ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);
        glm::vec2 guiSize = winSize * glm::vec2(0.4f, 0.55f);
        ImGui::SetNextWindowSize(ImVec2(guiSize.x, guiSize.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 60), ImGuiCond_FirstUseEver);
        flags = 0;
        ImGui::Begin(app.config.title.c_str(), 0, flags);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));

        ImGui::Separator();

        ImGui::Text("Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoTexture.getFrameRate(), 1000.0f / videoTexture.getFrameRate());

        ImGui::Separator();

        ImGui::Text("Pose URL: %s", poseURL.c_str());
        ImGui::Text("Input URL: %s", inputUrl.c_str());

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
        .fragmentCodePath = "shaders/displayVideo.frag"
    });

    app.onRender([&](double now, double dt){
        // handle mouse input
        ImGuiIO& io = ImGui::GetIO();
        if (!(io.WantCaptureKeyboard || io.WantCaptureMouse)) {
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
        }

        // handle keyboard input
        auto keys = window.getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window.close();
        }

        poseStreamer.sendPose();

        // render video
        screenShader.bind();
        screenShader.setInt("videoTexture", 4);
        videoTexture.bind(4);
        videoTexture.draw();

        // render to screen
        app.renderer.drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    videoTexture.cleanup();

    return 0;
}
