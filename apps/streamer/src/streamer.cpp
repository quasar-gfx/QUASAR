#include <iostream>

#include <imgui/imgui.h>

#include <Shader.h>
#include <Texture.h>
#include <Mesh.h>
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

void processInput(OpenGLApp* app, Camera* camera, float deltaTime);

const std::string BACKPACK_MODEL_PATH = "../../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Video Streamer";

    std::string outputUrl = "udp://localhost:1234";
    std::string modelPath = "../../assets/models/Sponza/Sponza.gltf";
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
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputUrl = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            app.config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
    }

    app.init();

    int screenWidth, screenHeight;
    app.getWindowSize(&screenWidth, &screenHeight);

    Scene* scene = new Scene();
    Camera* camera = new Camera(screenWidth, screenHeight);

    VideoStreamer videoStreamer = VideoStreamer();

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Text("Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamer.getFrameRate(), 1000.0f / videoStreamer.getFrameRate());
        ImGui::End();
    });

    app.onResize([&camera](unsigned int width, unsigned int height) {
        camera->aspect = (float)width / (float)height;
    });

    app.onMouseMove([&app, &camera](double xposIn, double yposIn) {
        static bool mouseDown = false;

        static float lastX = app.config.width / 2.0;
        static float lastY = app.config.height / 2.0;

        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

        lastX = xpos;
        lastY = ypos;

        if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            mouseDown = true;
            glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }

        if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            mouseDown = false;
            glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }

        if (mouseDown) {
            camera->processMouseMovement(xoffset, yoffset);
        }
    });

    app.onMouseScroll([&app, &camera](double xoffset, double yoffset) {
        camera->processMouseScroll(static_cast<float>(yoffset));
    });

    // shaders
    Shader pbrShader, screenShader;
    pbrShader.loadFromFile("shaders/pbr.vert", "shaders/pbr.frag");
    screenShader.loadFromFile("shaders/postprocess.vert", "shaders/postprocess.frag");

    // converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader;
    equirectToCubeMapShader.loadFromFile("shaders/skybox.vert", "shaders/equirectangular2cubemap.frag");

    // solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader;
    convolutionShader.loadFromFile("shaders/skybox.vert", "shaders/irradianceConvolution.frag");

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader;
    prefilterShader.loadFromFile("shaders/skybox.vert", "shaders/prefilter.frag");

    // BRDF shader
    Shader brdfShader;
    brdfShader.loadFromFile("shaders/brdf.vert", "shaders/brdf.frag");

    // background skybox shader
    Shader backgroundShader;
    backgroundShader.loadFromFile("shaders/background.vert", "shaders/background.frag");

    // textures
    Texture albedo = Texture("../../assets/textures/pbr/gold/albedo.png");
    Texture normal = Texture("../../assets/textures/pbr/gold/normal.png");
    Texture metallic = Texture("../../assets/textures/pbr/gold/metallic.png");
    Texture roughness = Texture("../../assets/textures/pbr/gold/roughness.png");
    Texture ao = Texture("../../assets/textures/pbr/gold/ao.png");
    std::vector<TextureID> goldTextures = { albedo.ID, 0, normal.ID, metallic.ID, roughness.ID, ao.ID };

    Texture ironAlbedo = Texture("../../assets/textures/pbr/rusted_iron/albedo.png");
    Texture ironNormal = Texture("../../assets/textures/pbr/rusted_iron/normal.png");
    Texture ironMetallic = Texture("../../assets/textures/pbr/rusted_iron/metallic.png");
    Texture ironRoughness = Texture("../../assets/textures/pbr/rusted_iron/roughness.png");
    Texture ironAo = Texture("../../assets/textures/pbr/rusted_iron/ao.png");
    std::vector<TextureID> ironTextures = { ironAlbedo.ID, 0, ironNormal.ID, ironMetallic.ID, ironRoughness.ID, ironAo.ID };

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
    Mesh* cubeMeshGold = new Mesh(cubeVertices, goldTextures);
    Node* cubeNodeGold = new Node(cubeMeshGold);
    cubeNodeGold->setTranslation(glm::vec3(-5.0f, 0.5f, -1.0f));

    Mesh* cubeMeshIron = new Mesh(cubeVertices, ironTextures);
    Node* cubeNodeIron = new Node(cubeMeshIron);
    cubeNodeIron->setTranslation(glm::vec3(5.0f, 0.5f, -1.0f));

    // lights
    DirectionalLight* directionalLight = new DirectionalLight(glm::vec3(0.8f, 0.8f, 0.8f), 10.0f);
    directionalLight->setDirection(glm::vec3(-0.2f, -1.0f, -0.3f));

    PointLight* pointLight1 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight1->setPosition(glm::vec3(-1.45f, 0.9f, -6.2f));
    pointLight1->setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight* pointLight2 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight2->setPosition(glm::vec3(2.2f, 0.9f, -6.2f));
    pointLight2->setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight* pointLight3 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight3->setPosition(glm::vec3(-1.45f, 0.9f, 4.89f));
    pointLight3->setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight* pointLight4 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight4->setPosition(glm::vec3(2.2f, 0.9f, 4.89f));
    pointLight4->setAttenuation(0.0f, 0.09f, 1.0f);

    // models
    Model* sponza = new Model(modelPath);

    Node* sponzaNode = new Node(sponza);
    sponzaNode->setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode->setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));
    sponzaNode->setScale(glm::vec3(0.01f));

    Model* backpack = new Model(BACKPACK_MODEL_PATH, true);

    Node* backpackNode = new Node(backpack);
    backpackNode->setTranslation(glm::vec3(-0.25f, 0.25f, -3.0f));
    backpackNode->setScale(glm::vec3(0.25f));

    // load the HDR environment map
    Texture hdrTexture = Texture("../../assets/textures/hdr/barcelona.hdr", GL_FLOAT, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR, true);

    // skybox
    CubeMap envCubeMap = CubeMap(512, 512, CUBE_MAP_HDR);

    scene->setDirectionalLight(directionalLight);
    scene->addPointLight(pointLight1);
    scene->addPointLight(pointLight2);
    scene->addPointLight(pointLight3);
    scene->addPointLight(pointLight4);
    scene->addChildNode(cubeNodeGold);
    scene->addChildNode(cubeNodeIron);
    scene->addChildNode(sponzaNode);
    scene->addChildNode(backpackNode);

    scene->equirectToCubeMap(envCubeMap, hdrTexture, equirectToCubeMapShader);
    scene->setupIBL(envCubeMap, convolutionShader, prefilterShader, brdfShader);
    scene->setEnvMap(&envCubeMap);

    int ret = videoStreamer.start(app.renderer.gBuffer.colorBuffer, outputUrl);
    if (ret < 0) {
        std::cerr << "Failed to initialize FFMpeg Video Streamer" << std::endl;
        return ret;
    }

    app.onRender([&](double now, double dt) {
        processInput(&app, camera, dt);

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

    // cleanup
    app.cleanup();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}

void processInput(OpenGLApp* app, Camera* camera, float deltaTime) {
    if (glfwGetKey(app->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(app->window, true);

    if (glfwGetKey(app->window, GLFW_KEY_W) == GLFW_PRESS)
        camera->processKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_S) == GLFW_PRESS)
        camera->processKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_A) == GLFW_PRESS)
        camera->processKeyboard(LEFT, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_D) == GLFW_PRESS)
        camera->processKeyboard(RIGHT, deltaTime);
}
