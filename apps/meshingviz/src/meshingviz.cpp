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

const std::string BACKPACK_MODEL_PATH = "../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Meshing Visualizer";

    std::string modelPath = "../meshing/mesh.obj";
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

    // shaders
    Shader pbrShader, screenShader;
    pbrShader.loadFromFile("../assets/shaders/simple.vert", "../assets/shaders/textured.frag");
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

    // Texture ironAlbedo = Texture("../assets/textures/pbr/rusted_iron/albedo.png");
    // Texture ironNormal = Texture("../assets/textures/pbr/rusted_iron/normal.png");
    // Texture ironMetallic = Texture("../assets/textures/pbr/rusted_iron/metallic.png");
    // Texture ironRoughness = Texture("../assets/textures/pbr/rusted_iron/roughness.png");
    // Texture ironAo = Texture("../assets/textures/pbr/rusted_iron/ao.png");
    // std::vector<TextureID> ironTextures = { ironAlbedo.ID, 0, ironNormal.ID, ironMetallic.ID, ironRoughness.ID, ironAo.ID };

    // Texture plasticAlbedo = Texture("../assets/textures/pbr/plastic/albedo.png");
    // Texture plasticNormal = Texture("../assets/textures/pbr/plastic/normal.png");
    // Texture plasticMetallic = Texture("../assets/textures/pbr/plastic/metallic.png");
    // Texture plasticRoughness = Texture("../assets/textures/pbr/plastic/roughness.png");
    // Texture plasticAo = Texture("../assets/textures/pbr/plastic/ao.png");
    // std::vector<TextureID> plasticTextures = { plasticAlbedo.ID, 0, plasticNormal.ID, plasticMetallic.ID, plasticRoughness.ID, plasticAo.ID };

    // Texture windowTexture = Texture("../assets/textures/window.png");
    // std::vector<TextureID> windowTextures = { windowTexture.ID };

    // objects
    // Cube cubeGold = Cube(goldTextures);
    // Node cubeNodeGold = Node(cubeGold);
    // cubeNodeGold.setTranslation(glm::vec3(-0.2f, 0.25f, -7.0f));
    // cubeNodeGold.setScale(glm::vec3(0.5f));

    // Cube cubeIron = Cube(ironTextures);
    // Node cubeNodeIron = Node(cubeIron);
    // cubeNodeIron.setTranslation(glm::vec3(1.5f, 0.25f, -3.0f));
    // cubeNodeIron.setScale(glm::vec3(0.5f));

    // Sphere sphere = Sphere(plasticTextures);
    // Node sphereNodePlastic = Node(sphere);
    // sphereNodePlastic.setTranslation(glm::vec3(1.0f, 1.5f, -8.0f));
    // sphereNodePlastic.setScale(glm::vec3(0.5f));

    // Plane plane = Plane(windowTextures);
    // Node planeNode = Node(plane);
    // planeNode.setTranslation(glm::vec3(0.0f, 1.5f, -7.0f));
    // planeNode.setRotationEuler(glm::vec3(-90.0f, 0.0f, 0.0f));
    // planeNode.setScale(glm::vec3(0.5f));

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

    textureParams.path = "../meshing/meshColor.png";
    Texture meshTexture = Texture(textureParams);
    std::vector<TextureID> meshTextures = { meshTexture.ID };
    // Cube cube = Cube(meshTextures);
    // Node cubeNode = Node(&cube);

    // models
    ModelCreateParams modelParams{
        .path = modelPath,
        .inputTextures = meshTextures
    };
    Model mesh = Model(modelParams);
    Node meshNode = Node(&mesh);

    // Model backpack = Model(BACKPACK_MODEL_PATH, true);

    // Node backpackNode = Node(&backpack);
    // backpackNode.setTranslation(glm::vec3(0.5f, 0.1f, -5.0f));
    // backpackNode.setScale(glm::vec3(0.25f));

    scene.setDirectionalLight(&directionalLight);
    scene.addPointLight(&pointLight1);
    scene.addPointLight(&pointLight2);
    scene.addPointLight(&pointLight3);
    scene.addPointLight(&pointLight4);
    // scene.addChildNode(&cubeNodeGold);
    // scene.addChildNode(&cubeNodeIron);
    // scene.addChildNode(sphereNodePlastic);
    scene.addChildNode(&meshNode);
    // scene.addChildNode(&cubeNode);
    // scene.addChildNode(&backpackNode);
    // scene.addChildNode(&planeNode);

    // Shader dirLightShadowsShader;
    // dirLightShadowsShader.loadFromFile("../assets/shaders/shadows/dirShadow.vert", "../assets/shaders/shadows/dirShadow.frag");

    // Shader pointLightShadowsShader;
    // pointLightShadowsShader.loadFromFile("../assets/shaders/shadows/pointShadow.vert", "../assets/shaders/shadows/pointShadow.frag", "../assets/shaders/shadows/pointShadow.geo");

    app.onRender([&](double now, double dt) {
        // handle mouse buttons
        auto mouseButtons = window.getMouseButtons();
        window.setMouseCursor(!mouseButtons.LEFT_PRESSED);
        if (mouseButtons.LEFT_PRESSED) {
            static bool firstMouse = true;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;

            auto cursorPos = window.getCursorPos();
            float xpos = static_cast<float>(cursorPos.x);
            float ypos = static_cast<float>(cursorPos.y);

            if (firstMouse) {
                lastX = xpos;
                lastY = ypos;
                firstMouse = false;
            }

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

        // app.renderer.updateDirLightShadowMap(dirLightShadowsShader, scene, camera);
        // app.renderer.updatePointLightShadowMaps(pointLightShadowsShader, scene, camera);

        // render all objects in scene
        app.renderer.drawObjects(pbrShader, scene, camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(backgroundShader, scene, camera);

        // render to screen
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
