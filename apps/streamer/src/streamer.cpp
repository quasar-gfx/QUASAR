#include <iostream>

#include <imgui/imgui.h>

#include <Shader.h>
#include <Texture.h>
#include <Primatives.h>
#include <Model.h>
#include <CubeMap.h>
#include <Entity.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights.h>
#include <Framebuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <GLFWWindow.h>

#include <VideoStreamer.h>
#include <PoseReceiver.h>

const std::string BACKPACK_MODEL_PATH = "../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Video Streamer";

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
    }

    GLFWWindow window(app.config);
    app.init(&window);

    unsigned int screenWidth, screenHeight;
    window.getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

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

    // shaders
    Shader pbrShader({
        .vertexCodePath = "../assets/shaders/pbr/pbr.vert",
        .fragmentCodePath = "../assets/shaders/pbr/pbr.frag"
    });

    // converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader({
        .vertexCodePath = "../assets/shaders/cubemap/cubemap.vert",
        .fragmentCodePath = "../assets/shaders/cubemap/equirectangular2cubemap.frag"
    });

    // solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader({
        .vertexCodePath = "../assets/shaders/cubemap/cubemap.vert",
        .fragmentCodePath = "../assets/shaders/pbr/irradianceConvolution.frag"
    });

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader({
        .vertexCodePath = "../assets/shaders/cubemap/cubemap.vert",
        .fragmentCodePath = "../assets/shaders/pbr/prefilter.frag"
    });

    // BRDF shader
    Shader brdfShader({
        .vertexCodePath = "../assets/shaders/pbr/brdf.vert",
        .fragmentCodePath = "../assets/shaders/pbr/brdf.frag"
    });

    // background skybox shader
    Shader backgroundShader({
        .vertexCodePath = "../assets/shaders/cubemap/background.vert",
        .fragmentCodePath = "../assets/shaders/cubemap/backgroundHDR.frag"
    });

    Shader dirLightShadowsShader({
        .vertexCodePath = "../assets/shaders/shadows/dirShadow.vert",
        .fragmentCodePath = "../assets/shaders/shadows/dirShadow.frag"
    });

    Shader pointLightShadowsShader({
        .vertexCodePath = "../assets/shaders/shadows/pointShadow.vert",
        .fragmentCodePath = "../assets/shaders/shadows/pointShadow.frag",
        .geometryCodePath = "../assets/shaders/shadows/pointShadow.geom"
    });

    Shader screenShader({
        .vertexCodePath = "../assets/shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../assets/shaders/postprocessing/displayColor.frag"
    });

    // materials
    Material goldMaterial = Material({
        .albedoTexturePath = "../assets/textures/pbr/gold/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/gold/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/gold/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/gold/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/gold/ao.png"
    });

    Material ironMaterial = Material({
        .albedoTexturePath = "../assets/textures/pbr/rusted_iron/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/rusted_iron/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/rusted_iron/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/rusted_iron/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/rusted_iron/ao.png"
    });

    Material plasticMaterial = Material({
        .albedoTexturePath = "../assets/textures/pbr/plastic/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/plastic/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/plastic/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/plastic/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/plastic/ao.png"
    });

    Material windowMaterial = Material({
        .albedoTexturePath = "../assets/textures/window.png"
    });

    // objects
    Cube cubeGold = Cube(goldMaterial);
    Node cubeNodeGold = Node(&cubeGold);
    cubeNodeGold.setTranslation(glm::vec3(-0.2f, 0.25f, -7.0f));
    cubeNodeGold.setScale(glm::vec3(0.5f));

    Cube cubeIron = Cube(ironMaterial);
    Node cubeNodeIron = Node(&cubeIron);
    cubeNodeIron.setTranslation(glm::vec3(1.5f, 0.25f, -3.0f));
    cubeNodeIron.setScale(glm::vec3(0.5f));

    Sphere sphere = Sphere(plasticMaterial);
    Node sphereNodePlastic = Node(&sphere);
    sphereNodePlastic.setTranslation(glm::vec3(1.0f, 1.5f, -8.0f));
    sphereNodePlastic.setScale(glm::vec3(0.5f));

    Plane plane = Plane(windowMaterial);
    Node planeNode = Node(&plane);
    planeNode.setTranslation(glm::vec3(0.0f, 1.5f, -7.0f));
    planeNode.setRotationEuler(glm::vec3(-90.0f, 0.0f, 0.0f));
    planeNode.setScale(glm::vec3(0.5f));

    // lights
    DirectionalLight directionalLight = DirectionalLight({
        .color = glm::vec3(0.8f, 0.8f, 0.8f),
        .direction = glm::vec3(0.0f, -1.0f, -0.3f),
        .intensity = 0.1f
    });

    PointLight pointLight1 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(-1.45f, 3.5f, -6.2f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight2 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(2.2f, 3.5f, -6.2f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight3 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(-1.45f, 3.5f, 4.89f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight4 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(2.2f, 3.5f, 4.89f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    // models
    Model sponza = Model({ .path = modelPath });
    Node sponzaNode = Node(&sponza);
    sponzaNode.setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode.setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));

    Model backpack = Model({ .path = BACKPACK_MODEL_PATH, .flipTextures = true });
    Node backpackNode = Node(&backpack);
    backpackNode.setTranslation(glm::vec3(0.5f, 0.1f, -5.0f));
    backpackNode.setScale(glm::vec3(0.25f));

    // load the HDR environment map
    Texture hdrTexture = Texture({
        .internalFormat = GL_RGB16F,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .flipped = true,
        .path = hdrImagePath
    });

    // skybox
    CubeMap envCubeMap({ .width = 512, .height = 512, .type = CUBE_MAP_HDR });

    scene.setDirectionalLight(&directionalLight);
    scene.addPointLight(&pointLight1);
    scene.addPointLight(&pointLight2);
    scene.addPointLight(&pointLight3);
    scene.addPointLight(&pointLight4);
    scene.addChildNode(&cubeNodeGold);
    scene.addChildNode(&cubeNodeIron);
    scene.addChildNode(&sphereNodePlastic);
    scene.addChildNode(&sponzaNode);
    scene.addChildNode(&backpackNode);
    scene.addChildNode(&planeNode);

    scene.equirectToCubeMap(envCubeMap, hdrTexture, equirectToCubeMapShader);
    scene.setupIBL(envCubeMap, convolutionShader, prefilterShader, brdfShader);
    scene.setEnvMap(&envCubeMap);

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

        app.renderer.updateDirLightShadowMap(dirLightShadowsShader, scene, camera);
        app.renderer.updatePointLightShadowMaps(pointLightShadowsShader, scene, camera);

        // animate lights
        cubeNodeGold.setRotationEuler(glm::vec3(0.0f, 10.0f * now, 0.0f));
        pointLight1.setPosition(glm::vec3(-1.45f + 0.25f * sin(now), 3.5f, -6.2f + 0.25f * cos(now)));
        pointLight2.setPosition(glm::vec3(2.2f + 0.25f * sin(now), 3.5f, -6.2f + 0.25f * cos(now)));
        pointLight3.setPosition(glm::vec3(-1.45f + 0.25f * sin(now), 3.5f, 4.89f + 0.25f * cos(now)));
        pointLight4.setPosition(glm::vec3(2.2f + 0.25f * sin(now), 3.5f, 4.89f + 0.25f * cos(now)));

        // render all objects in scene
        app.renderer.drawObjects(pbrShader, scene, camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(backgroundShader, scene, camera);

        // render to screen
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);

        videoStreamer.sendFrame();
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
