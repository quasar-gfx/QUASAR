#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Materials/UnlitMaterial.h>
#include <Materials/PBRMaterial.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#define VERTICES_IN_A_QUAD 4

const std::string DATA_PATH = "../quads/data/";

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int surfelSize = 4;
RenderState renderState = RenderState::MESH;

int createMesh(Mesh* mesh, std::string label) {
    std::ifstream vertexFile(DATA_PATH + "positions_" + label + "_0.bin", std::ios::binary);
    if (!vertexFile.is_open()) {
        std::cerr << "Failed to open file with label=" << label << std::endl;
        return -1;
    }

    std::ifstream indexFile(DATA_PATH + "indices_" + label + "_0.bin", std::ios::binary);
    if (!indexFile.is_open()) {
        std::cerr << "Failed to open file with label=" << label << std::endl;
        return -1;
    }

    Texture diffuseTexture = Texture({
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR,
        .path = "../quads/imgs/color_" + label + "_0.png"
    });

    unsigned int width = diffuseTexture.width / surfelSize;
    unsigned int height = diffuseTexture.height / surfelSize;

    unsigned int x = 0, y = 0;
    std::vector<Vertex> vertices;
    for (int i = 0; vertexFile; i+=VERTICES_IN_A_QUAD) {
        x = (i / VERTICES_IN_A_QUAD) % width;
        y = (i / VERTICES_IN_A_QUAD) / width;

        Vertex vertexUpperLeft;
        vertexFile.read(reinterpret_cast<char*>(&vertexUpperLeft.position), sizeof(glm::vec3));
        vertexUpperLeft.texCoords = glm::vec2((float)x / (float)(width), 1.0f - (float)(y + 1) / (float)(height));

        Vertex vertexUpperRight;
        vertexFile.read(reinterpret_cast<char*>(&vertexUpperRight.position), sizeof(glm::vec3));
        vertexUpperRight.texCoords = glm::vec2((float)(x + 1) / (float)(width), 1.0f - (float)(y + 1) / (float)(height));

        Vertex vertexLowerLeft;
        vertexFile.read(reinterpret_cast<char*>(&vertexLowerLeft.position), sizeof(glm::vec3));
        vertexLowerLeft.texCoords = glm::vec2((float)x / (float)(width), 1.0f - (float)y / (float)(height));

        Vertex vertexLowerRight;
        vertexFile.read(reinterpret_cast<char*>(&vertexLowerRight.position), sizeof(glm::vec3));
        vertexLowerRight.texCoords = glm::vec2((float)(x + 1) / (float)(width), 1.0f - (float)y / (float)(height));

        vertices.push_back(vertexUpperLeft);
        vertices.push_back(vertexUpperRight);
        vertices.push_back(vertexLowerLeft);
        vertices.push_back(vertexLowerRight);
    }
    vertexFile.close();

    std::vector<unsigned int> indices;
    while (indexFile) {
        unsigned int index;
        indexFile.read(reinterpret_cast<char*>(&index), sizeof(unsigned int));
        indices.push_back(index);
    }
    indexFile.close();

    bool renderWireframe = renderState == RenderState::WIREFRAME;
    bool renderPointcloud = renderState == RenderState::POINTCLOUD;

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
    Config config{};
    config.title = "Quads Visualizer";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-ss") && i + 1 < argc) {
            surfelSize = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-pc") && i + 1 < argc) {
            renderState = RenderState::POINTCLOUD;
            i++;
        }
        else if (!strcmp(argv[i], "-wf") && i + 1 < argc) {
            renderState = RenderState::WIREFRAME;
            i++;
        }
    }

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

    int numVertices = 0;
    guiManager->onRender([&](double now, double dt) {
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        int flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("", 0, flags);
        ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);
        glm::vec2 guiSize = winSize * glm::vec2(0.4f, 0.3f);
        ImGui::SetNextWindowSize(ImVec2(guiSize.x, guiSize.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 60), ImGuiCond_FirstUseEver);
        flags = 0;
        ImGui::Begin(config.title.c_str(), 0, flags);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));

        ImGui::Separator();

        ImGui::Text("Number of vertices: %d", numVertices);

        ImGui::Separator();

        ImGui::InputFloat3("Camera Position", (float*)&camera.position);
        ImGui::SliderFloat("Movement speed", &camera.movementSpeed, 0.1f, 20.0f);

        ImGui::Separator();

        ImGui::RadioButton("Render Mesh", (int*)&renderState, 0);
        ImGui::RadioButton("Render Point Cloud", (int*)&renderState, 1);
        ImGui::RadioButton("Render Wireframe", (int*)&renderState, 2);
        ImGui::End();
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    // shaders
    Shader screenShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });
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

    std::cout << numVertices << " vertices" << std::endl;

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    // load camera view and projection matrices
    std::ifstream cameraFile(DATA_PATH + "camera.bin", std::ios::binary);
    glm::mat4 proj = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    cameraFile.read(reinterpret_cast<char*>(&proj), sizeof(glm::mat4));
    cameraFile.read(reinterpret_cast<char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    camera.setProjectionMatrix(proj);
    camera.setViewMatrix(view);

    app.onRender([&](double now, double dt) {
        // handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;
            if (!prevMouseLeftPressed && mouseButtons.LEFT_PRESSED) {
                dragging = true;
                prevMouseLeftPressed = true;

                auto cursorPos = window->getCursorPos();
                lastX = static_cast<float>(cursorPos.x);
                lastY = static_cast<float>(cursorPos.y);
            }
            if (prevMouseLeftPressed && !mouseButtons.LEFT_PRESSED) {
                dragging = false;
                prevMouseLeftPressed = false;
            }
            if (dragging) {
                auto cursorPos = window->getCursorPos();
                float xpos = static_cast<float>(cursorPos.x);
                float ypos = static_cast<float>(cursorPos.y);

                float xoffset = xpos - lastX;
                float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

                lastX = xpos;
                lastY = ypos;

                camera.processMouseMovement(xoffset, yoffset, true);
            }

            // handle keyboard input
            auto keys = window->getKeys();
            camera.processKeyboard(keys, dt);
            if (keys.ESC_PRESSED) {
                window->close();
            }
        }

        for (auto& mesh : meshes) {
            if (renderState == RenderState::MESH) {
                mesh.pointcloud = false;
                mesh.wireframe = false;
            }
            else if (renderState == RenderState::POINTCLOUD) {
                mesh.pointcloud = true;
                mesh.wireframe = false;
            }
            else if (renderState == RenderState::WIREFRAME) {
                mesh.pointcloud = false;
                mesh.wireframe = true;
            }
        }

        // render all objects in scene
        app.renderer->drawObjects(scene, camera);

        // render to screen
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
