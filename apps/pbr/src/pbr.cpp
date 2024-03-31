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

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "PBR";

    std::string modelPath = "../assets/models/cerberus/cerberus.fbx";
    std::string hdrImagePath = "../assets/textures/hdr/newport_loft.hdr";
    std::string cube1TexturePath = "../assets/textures/pbr/gold";
    std::string cube2TexturePath = "../assets/textures/pbr/rusted_iron";
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
        else if (!strcmp(argv[i], "-t1") && i + 1 < argc) {
            cube1TexturePath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-t2") && i + 1 < argc) {
            cube2TexturePath = argv[i + 1];
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

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();
    });

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
    textureParams.path = cube1TexturePath + "/albedo.png";
    Texture albedo = Texture(textureParams);
    textureParams.path = cube1TexturePath + "/normal.png";
    Texture normal = Texture(textureParams);
    textureParams.path = cube1TexturePath + "/metallic.png";
    Texture metallic = Texture(textureParams);
    textureParams.path = cube1TexturePath + "/roughness.png";
    Texture roughness = Texture(textureParams);
    textureParams.path = cube1TexturePath + "/ao.png";
    Texture ao = Texture(textureParams);
    std::vector<TextureID> goldTextures = { albedo.ID, 0, normal.ID, metallic.ID, roughness.ID, ao.ID };

    textureParams.path = cube2TexturePath + "/albedo.png";
    Texture ironAlbedo = Texture(textureParams);
    textureParams.path = cube2TexturePath + "/normal.png";
    Texture ironNormal = Texture(textureParams);
    textureParams.path = cube2TexturePath + "/metallic.png";
    Texture ironMetallic = Texture(textureParams);
    textureParams.path = cube2TexturePath + "/roughness.png";
    Texture ironRoughness = Texture(textureParams);
    textureParams.path = cube2TexturePath + "/ao.png";
    Texture ironAo = Texture(textureParams);
    std::vector<TextureID> ironTextures = { ironAlbedo.ID, 0, ironNormal.ID, ironMetallic.ID, ironRoughness.ID, ironAo.ID };

    Sphere sphereGold = Sphere(goldTextures);
    Node sphereNodeGold = Node(&sphereGold);
    sphereNodeGold.setTranslation(glm::vec3(-5.0f, 0.5f, -1.0f));

    Cube cubeIron = Cube(ironTextures);
    Node cubeNodeIron = Node(&cubeIron);
    cubeNodeIron.setTranslation(glm::vec3(5.0f, 0.5f, -1.0f));

    textureParams.path = "../assets/models/cerberus/Textures/Cerberus_A.tga";
    Texture gunAlbedo = Texture(textureParams);
    textureParams.path = "../assets/models/cerberus/Textures/Cerberus_N.tga";
    Texture gunNormal = Texture(textureParams);
    textureParams.path = "../assets/models/cerberus/Textures/Cerberus_M.tga";
    Texture gunMetallic = Texture(textureParams);
    textureParams.path = "../assets/models/cerberus/Textures/Cerberus_R.tga";
    Texture gunRoughness = Texture(textureParams);
    textureParams.path = "../assets/models/cerberus/Textures/Cerberus_AO.tga";
    Texture gunAo = Texture(textureParams);
    std::vector<TextureID> gunTextures = { gunAlbedo.ID, 0, gunNormal.ID, gunMetallic.ID, gunRoughness.ID, gunAo.ID };

    // models
    ModelCreateParams gunParams{
        .path = modelPath,
        .inputTextures = gunTextures
    };
    Model gun = Model(gunParams);
    Node gunNode = Node(&gun);
    gunNode.setTranslation(glm::vec3(2.0f, 1.0f, -1.0f));
    gunNode.setRotationEuler(glm::vec3(0.0f, 90.0f, 0.0f));
    gunNode.setScale(glm::vec3(0.05f));

    // load the HDR environment map
    TextureCreateParams hdrTextureParams{
        .internalFormat = GL_RGB16F,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .flipped = true,
        .path = hdrImagePath
    };
    Texture hdrTexture = Texture(hdrTextureParams);

    // skybox
    CubeMap envCubeMap(512, 512, CUBE_MAP_HDR);

    // lights
    PointLight pointLight1 = PointLight(glm::vec3(1.0, 1.0, 1.0), 300.0f);
    pointLight1.setPosition(glm::vec3(-10.0f, 10.0f, 10.0f));
    pointLight1.setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight pointLight2 = PointLight(glm::vec3(1.0, 1.0, 1.0), 300.0f);
    pointLight2.setPosition(glm::vec3(10.0f, 10.0f, 10.0f));
    pointLight2.setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight pointLight3 = PointLight(glm::vec3(1.0, 1.0, 1.0), 300.0f);
    pointLight3.setPosition(glm::vec3(-10.0f, -10.0f, 10.0f));
    pointLight3.setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight pointLight4 = PointLight(glm::vec3(1.0, 1.0, 1.0), 300.0f);
    pointLight4.setPosition(glm::vec3(10.0f, -10.0f, 10.0f));
    pointLight4.setAttenuation(0.0f, 0.09f, 1.0f);

    scene.addPointLight(&pointLight1);
    scene.addPointLight(&pointLight2);
    scene.addPointLight(&pointLight3);
    scene.addPointLight(&pointLight4);
    scene.addChildNode(&sphereNodeGold);
    scene.addChildNode(&cubeNodeIron);
    scene.addChildNode(&gunNode);

    scene.equirectToCubeMap(envCubeMap, hdrTexture, equirectToCubeMapShader);
    scene.setupIBL(envCubeMap, convolutionShader, prefilterShader, brdfShader);
    scene.setEnvMap(&envCubeMap);

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
