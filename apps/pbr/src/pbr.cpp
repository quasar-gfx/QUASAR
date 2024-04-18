#include <iostream>

#include <imgui/imgui.h>

#include <Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Model.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <Framebuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <shaders.h>

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
    Shader pbrShader({
        .vertexData = SHADER_PBR_VERT,
        .vertexDataSize = SHADER_PBR_VERT_len,
        .fragmentData = SHADER_PBR_FRAG,
        .fragmentDataSize = SHADER_PBR_FRAG_len
    });

    // converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader({
        .vertexData = SHADER_CUBEMAP_VERT,
        .vertexDataSize = SHADER_CUBEMAP_VERT_len,
        .fragmentData = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG,
        .fragmentDataSize = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG_len
    });

    // solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader({
        .vertexData = SHADER_CUBEMAP_VERT,
        .vertexDataSize = SHADER_CUBEMAP_VERT_len,
        .fragmentData = SHADER_IRRADIANCECONVOLUTION_FRAG,
        .fragmentDataSize = SHADER_IRRADIANCECONVOLUTION_FRAG_len
    });

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader({
        .vertexData = SHADER_CUBEMAP_VERT,
        .vertexDataSize = SHADER_CUBEMAP_VERT_len,
        .fragmentData = SHADER_PREFILTER_FRAG,
        .fragmentDataSize = SHADER_PREFILTER_FRAG_len
    });

    // BRDF shader
    Shader brdfShader({
        .vertexData = SHADER_BRDF_VERT,
        .vertexDataSize = SHADER_BRDF_VERT_len,
        .fragmentData = SHADER_BRDF_FRAG,
        .fragmentDataSize = SHADER_BRDF_FRAG_len
    });

    // background skybox shader
    Shader backgroundShader({
        .vertexData = SHADER_BACKGROUND_VERT,
        .vertexDataSize = SHADER_BACKGROUND_VERT_len,
        .fragmentData = SHADER_BACKGROUNDHDR_FRAG,
        .fragmentDataSize = SHADER_BACKGROUNDHDR_FRAG_len
    });

    Shader screenShader({
        .vertexData = SHADER_POSTPROCESS_VERT,
        .vertexDataSize = SHADER_POSTPROCESS_VERT_len,
        .fragmentData = SHADER_DISPLAYCOLOR_FRAG,
        .fragmentDataSize = SHADER_DISPLAYCOLOR_FRAG_len
    });

    // materials
    PBRMaterial goldMaterial = PBRMaterial({
        .albedoTexturePath = "../assets/textures/pbr/gold/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/gold/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/gold/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/gold/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/gold/ao.png"
    });

    PBRMaterial ironMaterial = PBRMaterial({
        .albedoTexturePath = "../assets/textures/pbr/rusted_iron/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/rusted_iron/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/rusted_iron/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/rusted_iron/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/rusted_iron/ao.png"
    });

    PBRMaterial gunMaterial = PBRMaterial({
        .albedoTexturePath = "../assets/models/cerberus/Textures/Cerberus_A.tga",
        .normalTexturePath = "../assets/models/cerberus/Textures/Cerberus_N.tga",
        .metallicTexturePath = "../assets/models/cerberus/Textures/Cerberus_M.tga",
        .roughnessTexturePath = "../assets/models/cerberus/Textures/Cerberus_R.tga",
        .aoTexturePath = "../assets/models/cerberus/Textures/Cerberus_AO.tga"
    });

    // objects
    Sphere sphereGold = Sphere(&goldMaterial);
    Node sphereNodeGold = Node(&sphereGold);
    sphereNodeGold.setTranslation(glm::vec3(-5.0f, 0.5f, -1.0f));

    Cube cubeIron = Cube(&ironMaterial);
    Node cubeNodeIron = Node(&cubeIron);
    cubeNodeIron.setTranslation(glm::vec3(5.0f, 0.5f, -1.0f));

    // models
    Model gun = Model({
        .path = modelPath,
        .material = &gunMaterial
    });
    Node gunNode = Node(&gun);
    gunNode.setTranslation(glm::vec3(2.0f, 1.0f, -1.0f));
    gunNode.setRotationEuler(glm::vec3(0.0f, 90.0f, 0.0f));
    gunNode.setScale(glm::vec3(0.05f));

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

    // lights
    PointLight pointLight1 = PointLight({
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .initialPosition = glm::vec3(-10.0f, 10.0f, 10.0f),
        .intensity = 300.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight2 = PointLight({
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .initialPosition = glm::vec3(10.0f, 10.0f, 10.0f),
        .intensity = 300.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight3 = PointLight({
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .initialPosition = glm::vec3(-10.0f, -10.0f, 10.0f),
        .intensity = 300.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight4 = PointLight({
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .initialPosition = glm::vec3(10.0f, -10.0f, 10.0f),
        .intensity = 300.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

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
