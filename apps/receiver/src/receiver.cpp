#include <iostream>
#include <thread>

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
    app.config.title = "Video Receiver";
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

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Text("Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoTexture.getFrameRate(), 1000.0f / videoTexture.getFrameRate());
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
        .vertexCodeData = SHADER_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_DISPLAYVIDEO_FRAG,
        .fragmentCodeSize = SHADER_DISPLAYVIDEO_FRAG_len
    });

    // lights
    AmbientLight ambientLight = AmbientLight({ .color = glm::vec3(1.0f), .intensity = 1.0f });

    // materials
    DiffSpecMaterial containerMaterial = DiffSpecMaterial({ CONTAINER_TEXTURE });
    DiffSpecMaterial floorMaterial = DiffSpecMaterial({ METAL_TEXTURE });

    Cube cube = Cube({ .material = &containerMaterial });

    Node cubeNode = Node(&cube);
    cubeNode.setTranslation(glm::vec3(-1.0f, 0.0f, -1.0f));

    Sphere sphere = Sphere({ .material = &containerMaterial });

    Node sphereNode = Node(&sphere);
    sphereNode.setTranslation(glm::vec3(2.0f, 1.0f, -3.0f));

    Plane plane = Plane({ .material = &floorMaterial });
    Node planeNode = Node(&plane);
    planeNode.setScale(glm::vec3(25.0f, 1.0f, 25.0f));

    CubeMap skybox = CubeMap({
        .rightFaceTexturePath = "../assets/textures/skybox/right.jpg",
        .leftFaceTexturePath = "../assets/textures/skybox/left.jpg",
        .topFaceTexturePath = "../assets/textures/skybox/top.jpg",
        .bottomFaceTexturePath = "../assets/textures/skybox/bottom.jpg",
        .frontFaceTexturePath = "../assets/textures/skybox/front.jpg",
        .backFaceTexturePath = "../assets/textures/skybox/back.jpg"
    });

    scene.setAmbientLight(&ambientLight);
    scene.setEnvMap(&skybox);
    scene.addChildNode(&cubeNode);
    scene.addChildNode(&sphereNode);
    scene.addChildNode(&planeNode);

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

        poseStreamer.sendPose();

        // render all objects in scene
        app.renderer.drawObjects(scene, camera);

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
