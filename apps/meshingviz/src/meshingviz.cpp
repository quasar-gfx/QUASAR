#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Materials/UnlitMaterial.h>
#include <Materials/PBRMaterial.h>
#include <Primatives/Model.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>

int surfelSize = 4;
bool renderPointcloud = false;
bool renderWireframe = false;

int createMesh(Mesh* mesh, std::string label) {
    std::ifstream vertexFile("../meshing/data/positions_" + label + "_0.bin", std::ios::binary);
    if (!vertexFile.is_open()) {
        std::cerr << "Failed to open file with label=" << label << std::endl;
        return -1;
    }

    std::ifstream indexFile("../meshing/data/indices_" + label + "_0.bin", std::ios::binary);
    if (!indexFile.is_open()) {
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

    unsigned int width = diffuseTexture.width / surfelSize;
    unsigned int height = diffuseTexture.height / surfelSize;
    unsigned int idx = 0;
    std::vector<Vertex> vertices;
    for (int i = 0; i < width * height; i++) {
        Vertex vertex;
        vertexFile.read(reinterpret_cast<char*>(&vertex.position), sizeof(glm::vec3));
        // std::cout << vertex.position.x << " " << vertex.position.y << " " << vertex.position.z << std::endl;
        vertex.texCoords = glm::vec2((idx % width) / (float)(width - 1), 1.0f - (idx / width) / (float)(height - 1));
        idx++;
        vertices.push_back(vertex);
    }
    vertexFile.close();

    std::vector<unsigned int> indices;
    while (indexFile) {
        unsigned int index;
        indexFile.read(reinterpret_cast<char*>(&index), sizeof(unsigned int));
        indices.push_back(index);
    }
    indexFile.close();

    *mesh = Mesh({
        .vertices = vertices,
        .indices = indices,
        .material = new UnlitMaterial({ .diffuseTextureID = diffuseTexture.ID }),
        .wireframe = renderWireframe,
        .pointcloud = renderPointcloud,
    });

    return vertices.size();
}

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Meshing Visualizer";

    std::string modelPath = "../meshing/mesh.obj";
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
        else if (!strcmp(argv[i], "-wf") && i + 1 < argc) {
            renderWireframe = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-ss") && i + 1 < argc) {
            surfelSize = atoi(argv[i + 1]);
            i++;
        }
    }

    GLFWWindow window(app.config);
    app.init(&window);

    unsigned int screenWidth, screenHeight;
    window.getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

    int numVertices = 0;

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Text("Number of vertices: %d", numVertices);
        ImGui::Checkbox("Render Point Cloud", &renderPointcloud);
        ImGui::Checkbox("Render Wireframe", &renderWireframe);
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

    // UnlitMaterial containerMaterial = UnlitMaterial({ "../assets/textures/container.jpg" });
    // Cube cube = Cube({ .material = &containerMaterial });
    // Node cubeNode = Node(&cube);
    // cubeNode.setScale(glm::vec3(0.02f, 0.02f, 0.02f));
    // scene.addChildNode(&cubeNode);

    // models
    // Cube cube = Cube(goldMaterial);
    // Model mesh = Model({
    //     .path = modelPath,
    //     .material = meshMaterial,
    //     .pointcloud = renderPointcloud
    // });

    std::vector<std::string> labels = {
        "center",
        // "top_right_front",
        // "top_right_back",
        // "top_left_front",
        // "top_left_back",
        // "bottom_right_front",
        // "bottom_right_back",
        // "bottom_left_front",
        // "bottom_left_back"
    };

    std::vector<Mesh> meshes(labels.size());
    std::vector<Node> nodes(labels.size());
    for (int i = 0; i < labels.size(); i++) {
        numVertices += createMesh(&meshes[i], labels[i]);
        nodes[i] = Node(&meshes[i]);
        scene.addChildNode(&nodes[i]);
    }

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    // load camera view and projection matrices
    std::ifstream cameraFile("../meshing/data/camera.bin", std::ios::binary);
    glm::mat4 proj = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    cameraFile.read(reinterpret_cast<char*>(&proj), sizeof(glm::mat4));
    cameraFile.read(reinterpret_cast<char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    camera.setProjectionMatrix(proj);
    camera.setViewMatrix(view);

    app.onRender([&](double now, double dt) {
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

        for (auto& mesh : meshes) {
            mesh.wireframe = renderWireframe;
            mesh.pointcloud = renderPointcloud;
        }

        // render all objects in scene
        app.renderer.drawObjects(scene, camera);

        // render to screen
        app.renderer.drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
