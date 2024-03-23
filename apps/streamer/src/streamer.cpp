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
#include <FrameBuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>

#include <VideoStreamer.h>
#include <PoseReceiver.h>

void processInput(OpenGLApp &app, Camera &camera, float deltaTime);

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

    app.init();

    unsigned int screenWidth, screenHeight;
    app.getWindowSize(&screenWidth, &screenHeight);

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

    app.onResize([&camera](unsigned int width, unsigned int height) {
        camera.aspect = (float)width / (float)height;
    });

    // app.onMouseMove([&](double xposIn, double yposIn) {
    //     static bool mouseDown = false;

    //     static float lastX = screenWidth / 2.0;
    //     static float lastY = screenHeight / 2.0;

    //     float xpos = static_cast<float>(xposIn);
    //     float ypos = static_cast<float>(yposIn);

    //     float xoffset = xpos - lastX;
    //     float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    //     lastX = xpos;
    //     lastY = ypos;

    //     if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
    //         mouseDown = true;
    //         glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    //     }

    //     if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
    //         mouseDown = false;
    //         glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    //     }

    //     if (mouseDown) {
    //         camera.processMouseMovement(xoffset, yoffset);
    //     }
    // });

    // app.onMouseScroll([&app, &camera](double xoffset, double yoffset) {
    //     camera.processMouseScroll(static_cast<float>(yoffset));
    // });

    // shaders
    Shader pbrShader, screenShader;
    pbrShader.loadFromFile("../assets/shaders/pbr/pbr.vert", "../assets/shaders/pbr/pbr.frag");
    screenShader.loadFromFile("../assets/shaders/postprocessing/postprocess.vert", "../assets/shaders/postprocessing/displayColor.frag");

    // converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader;
    equirectToCubeMapShader.loadFromFile("../assets/shaders/cubemap/cubemap.vert", "../assets/shaders/cubemap/equirectangular2cubemap.frag");

    // solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader;
    convolutionShader.loadFromFile("../assets/shaders/cubemap/cubemap.vert", "../assets/shaders/pbr/irradianceConvolution.frag");

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader;
    prefilterShader.loadFromFile("../assets/shaders/cubemap/cubemap.vert", "../assets/shaders/pbr/prefilter.frag");

    // BRDF shader
    Shader brdfShader;
    brdfShader.loadFromFile("../assets/shaders/pbr/brdf.vert", "../assets/shaders/pbr/brdf.frag");

    // background skybox shader
    Shader backgroundShader;
    backgroundShader.loadFromFile("../assets/shaders/cubemap/background.vert", "../assets/shaders/cubemap/backgroundHDR.frag");

    // textures
    TextureCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR
    };
    textureParams.path = "../assets/textures/pbr/gold/albedo.png";
    Texture albedo = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/gold/normal.png";
    Texture normal = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/gold/metallic.png";
    Texture metallic = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/gold/roughness.png";
    Texture roughness = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/gold/ao.png";
    Texture ao = Texture(textureParams);
    std::vector<TextureID> goldTextures = { albedo.ID, 0, normal.ID, metallic.ID, roughness.ID, ao.ID };

    textureParams.path = "../assets/textures/pbr/rusted_iron/albedo.png";
    Texture ironAlbedo = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/rusted_iron/normal.png";
    Texture ironNormal = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/rusted_iron/metallic.png";
    Texture ironMetallic = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/rusted_iron/roughness.png";
    Texture ironRoughness = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/rusted_iron/ao.png";
    Texture ironAo = Texture(textureParams);
    std::vector<TextureID> ironTextures = { ironAlbedo.ID, 0, ironNormal.ID, ironMetallic.ID, ironRoughness.ID, ironAo.ID };

    textureParams.path = "../assets/textures/pbr/plastic/albedo.png";
    Texture plasticAlbedo = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/plastic/normal.png";
    Texture plasticNormal = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/plastic/metallic.png";
    Texture plasticMetallic = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/plastic/roughness.png";
    Texture plasticRoughness = Texture(textureParams);
    textureParams.path = "../assets/textures/pbr/plastic/ao.png";
    Texture plasticAo = Texture(textureParams);
    std::vector<TextureID> plasticTextures = { plasticAlbedo.ID, 0, plasticNormal.ID, plasticMetallic.ID, plasticRoughness.ID, plasticAo.ID };

    textureParams.path = "../assets/textures/window.png";
    textureParams.wrapS = GL_REPEAT;
    textureParams.wrapT = GL_REPEAT;
    Texture windowTexture = Texture(textureParams);
    std::vector<TextureID> windowTextures = { windowTexture.ID };

    // objects
    std::vector<Vertex> cubeVertices {
        // Front face
        { {-1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Left
        { { 1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Right
        { { 1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Right
        { { 1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Right
        { {-1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Left
        { {-1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Left

        // Back face
        { { 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Right
        { {-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Left
        { {-1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Left
        { {-1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Left
        { { 1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Right
        { { 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Right

        // Left face
        { {-1.0f, -1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },  // Bottom Front
        { {-1.0f, -1.0f,  1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },  // Bottom Back
        { {-1.0f,  1.0f,  1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f} },  // Top Back
        { {-1.0f,  1.0f,  1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f} },  // Top Back
        { {-1.0f,  1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, -1.0f} },  // Top Front
        { {-1.0f, -1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },  // Bottom Front

        // Right face
        { { 1.0f, -1.0f,  1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },  // Bottom Front
        { { 1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },  // Bottom Back
        { { 1.0f,  1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },  // Top Back
        { { 1.0f,  1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },  // Top Back
        { { 1.0f,  1.0f,  1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },  // Top Front
        { { 1.0f, -1.0f,  1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },  // Bottom Front

        // Top face
        { {-1.0f,  1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Left
        { { 1.0f,  1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Right
        { { 1.0f,  1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Right
        { { 1.0f,  1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Right
        { {-1.0f,  1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Left
        { {-1.0f,  1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Left

        // Bottom face
        { {-1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Left
        { { 1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Right
        { { 1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Right
        { { 1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Right
        { {-1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Left
        { {-1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} }   // Bottom Left
    };
    Mesh cubeGold = Mesh(cubeVertices, goldTextures);
    Node cubeNodeGold = Node(&cubeGold);
    cubeNodeGold.setTranslation(glm::vec3(-0.2f, 0.25f, -7.0f));
    cubeNodeGold.setScale(glm::vec3(0.5f));

    Mesh cubeIron = Mesh(cubeVertices, ironTextures);
    Node cubeNodeIron = Node(&cubeIron);
    cubeNodeIron.setTranslation(glm::vec3(1.5f, 0.25f, -3.0f));
    cubeNodeIron.setScale(glm::vec3(0.5f));

    Sphere sphere = Sphere(plasticTextures);
    Node sphereNodePlastic = Node(&sphere);
    sphereNodePlastic.setTranslation(glm::vec3(1.0f, 1.5f, -8.0f));
    sphereNodePlastic.setScale(glm::vec3(0.5f));

    Plane plane = Plane(windowTextures);
    Node planeNode = Node(&plane);
    planeNode.setTranslation(glm::vec3(0.0f, 1.5f, -7.0f));
    planeNode.setRotationEuler(glm::vec3(-90.0f, 0.0f, 0.0f));
    planeNode.setScale(glm::vec3(0.5f));

    // lights
    DirectionalLight directionalLight = DirectionalLight(glm::vec3(0.8f, 0.8f, 0.8f), 0.1f);
    directionalLight.setDirection(glm::vec3(0.0f, -1.0f, -0.3f));

    PointLight pointLight1 = PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight1.setPosition(glm::vec3(-1.45f, 3.5f, -6.2f));
    pointLight1.setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight pointLight2 = PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight2.setPosition(glm::vec3(2.2f, 3.5f, -6.2f));
    pointLight2.setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight pointLight3 = PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight3.setPosition(glm::vec3(-1.45f, 3.5f, 4.89f));
    pointLight3.setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight pointLight4 = PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight4.setPosition(glm::vec3(2.2f, 3.5f, 4.89f));
    pointLight4.setAttenuation(0.0f, 0.09f, 1.0f);

    // models
    ModelCreateParams sponzaParams{
        .path = modelPath
    };
    Model sponza = Model(sponzaParams);
    Node sponzaNode = Node(&sponza);
    sponzaNode.setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode.setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));

    ModelCreateParams backpackParams{
        .path = BACKPACK_MODEL_PATH,
        .flipTextures = true
    };
    Model backpack = Model(backpackParams);
    Node backpackNode = Node(&backpack);
    backpackNode.setTranslation(glm::vec3(0.5f, 0.1f, -5.0f));
    backpackNode.setScale(glm::vec3(0.25f));

    // load the HDR environment map
    TextureCreateParams hdrParams{
        .path = hdrImagePath,
        .internalFormat = GL_RGB16F,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .flipped = true
    };
    Texture hdrTexture = Texture(hdrParams);

    // skybox
    CubeMap envCubeMap(512, 512, CUBE_MAP_HDR);

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

    Shader dirLightShadowsShader;
    dirLightShadowsShader.loadFromFile("../assets/shaders/shadows/dirShadow.vert", "../assets/shaders/shadows/dirShadow.frag");

    Shader pointLightShadowsShader;
    pointLightShadowsShader.loadFromFile("../assets/shaders/shadows/pointShadow.vert", "../assets/shaders/shadows/pointShadow.frag", "../assets/shaders/shadows/pointShadow.geo");

    app.onRender([&](double now, double dt) {
        processInput(app, camera, dt);

        poseReceiver.receivePose();

        app.renderer.updateDirLightShadowMap(dirLightShadowsShader, &scene, &camera);
        app.renderer.updatePointLightShadowMaps(pointLightShadowsShader, &scene, &camera);

        // animate lights
        cubeNodeGold.setRotationEuler(glm::vec3(0.0f, 10.0f * now, 0.0f));
        pointLight1.setPosition(glm::vec3(-1.45f + 0.25f * sin(now), 3.5f, -6.2f + 0.25f * cos(now)));
        pointLight2.setPosition(glm::vec3(2.2f + 0.25f * sin(now), 3.5f, -6.2f + 0.25f * cos(now)));
        pointLight3.setPosition(glm::vec3(-1.45f + 0.25f * sin(now), 3.5f, 4.89f + 0.25f * cos(now)));
        pointLight4.setPosition(glm::vec3(2.2f + 0.25f * sin(now), 3.5f, 4.89f + 0.25f * cos(now)));

        // render all objects in scene
        app.renderer.drawObjects(pbrShader, &scene, &camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(backgroundShader, &scene, &camera);

        // render to screen
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);

        videoStreamer.sendFrame();
    });

    // run app loop (blocking)
    app.run();

    // cleanup
    app.cleanup();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}

void processInput(OpenGLApp &app, Camera &camera, float deltaTime) {
    if (glfwGetKey(app.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(app.window, true);

    // if (glfwGetKey(app->window, GLFW_KEY_W) == GLFW_PRESS)
    //     camera.processKeyboard(FORWARD, deltaTime);
    // if (glfwGetKey(app->window, GLFW_KEY_S) == GLFW_PRESS)
    //     camera.processKeyboard(BACKWARD, deltaTime);
    // if (glfwGetKey(app->window, GLFW_KEY_A) == GLFW_PRESS)
    //     camera.processKeyboard(LEFT, deltaTime);
    // if (glfwGetKey(app->window, GLFW_KEY_D) == GLFW_PRESS)
    //     camera.processKeyboard(RIGHT, deltaTime);
}
