#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Materials/TexturedMaterial.h>
#include <Materials/PBRMaterial.h>
#include <Primatives/Model.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <Framebuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>

const std::string BACKPACK_MODEL_PATH = "../assets/models/backpack/backpack.obj";

int createMesh(Mesh* mesh, std::string label, bool renderPointcloud) {
    std::vector<Vertex> vertices;
    std::ifstream file("../meshing/data/positions_" + label + "_0.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file with label=" << label << std::endl;
        return -1;
    }

    Texture diffuseTexture = Texture({
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR,
        .path = "../meshing/imgs/color_" + label + "_0.png"
    });

    unsigned int width = diffuseTexture.width;
    unsigned int height = diffuseTexture.height;
    unsigned int idx = 0;
    while (file) {
        Vertex vertex;
        file.read(reinterpret_cast<char*>(&vertex.position), sizeof(glm::vec3));
        vertex.texCoords = glm::vec2((idx % width) / (float)(width - 1), 1.0f - (idx / width) / (float)(height - 1));
        idx++;
        vertices.push_back(vertex);
    }
    file.close();

    *mesh = Mesh({
        .vertices = vertices,
        .material = new TexturedMaterial({ .diffuseTextureID = diffuseTexture.ID }),
        .pointcloud = renderPointcloud
    });

    return 0;
}

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Meshing Visualizer";

    std::string modelPath = "../meshing/mesh.obj";
    bool renderPointcloud = false;
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
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            renderPointcloud = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            app.config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
    }

    GLFWWindow window(app.config);
    app.init(&window);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

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
    Shader screenShader({
        .vertexCodeData = SHADER_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_DISPLAYCOLOR_FRAG,
        .fragmentCodeSize = SHADER_DISPLAYCOLOR_FRAG_len
    });

    // textures
    PBRMaterial goldMaterial = PBRMaterial({
        .albedoTexturePath = "../assets/textures/pbr/gold/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/gold/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/gold/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/gold/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/gold/ao.png"
    });

    // lights
    DirectionalLight directionalLight = DirectionalLight({
        .color = glm::vec3(0.8f, 0.8f, 0.8f),
        .direction = glm::vec3(0.0f, -1.0f, -0.3f),
        .intensity = 1.0f
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
    // Cube cube = Cube(goldMaterial);
    // Model mesh = Model({
    //     .path = modelPath,
    //     .material = meshMaterial,
    //     .pointcloud = renderPointcloud
    // });

    std::vector<std::string> labels = {
        "center",
        "top_right_front",
        "top_right_back",
        "top_left_front",
        "top_left_back",
        "bottom_right_front",
        "bottom_right_back",
        "bottom_left_front",
        "bottom_left_back"
    };

    std::vector<Mesh> meshes(labels.size());
    std::vector<Node> nodes(labels.size());
    for (int i = 0; i < labels.size(); i++) {
        createMesh(&meshes[i], labels[i], renderPointcloud);
        nodes[i] = Node(&meshes[i]);
        scene.addChildNode(&nodes[i]);
    }

    scene.setDirectionalLight(&directionalLight);
    scene.addPointLight(&pointLight1);
    scene.addPointLight(&pointLight2);
    scene.addPointLight(&pointLight3);
    scene.addPointLight(&pointLight4);

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    glm::vec3 initialPosition = glm::vec3(0.0f, 1.6f, 0.0f);
    camera.position = initialPosition;

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
        app.renderer.drawObjects(scene, camera);

        // render to screen
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
