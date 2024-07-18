#include <iostream>

#include <args.hxx>
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
#define NUM_SUB_QUADS 4

const std::string DATA_PATH = "../streamer/";

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int surfelSize = 4;
RenderState renderState = RenderState::MESH;

int createMesh(Mesh &mesh, Mesh &wireframeMesh, std::string label) {
    std::ifstream vertexFile(DATA_PATH + "data/vertices_" + label + "_0.bin", std::ios::binary);
    if (!vertexFile.is_open()) {
        std::cerr << "Failed to open vertex file with label=" << label << std::endl;
        return -1;
    }

    std::ifstream indexFile(DATA_PATH + "data/indices_" + label + "_0.bin", std::ios::binary);
    if (!indexFile.is_open()) {
        std::cerr << "Failed to open index file with label=" << label << std::endl;
        return -1;
    }

    Texture colorTexture = Texture({
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .flipVertically = true,
        .path = DATA_PATH + "imgs/color_" + label + "_0.png"
    });

    unsigned int width = colorTexture.width / surfelSize;
    unsigned int height = colorTexture.height / surfelSize;

    int numVertices = width * height * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    std::vector<Vertex> vertices(numVertices);
    vertexFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(Vertex));
    vertexFile.close();

    int numTriangles = width * height * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;
    std::vector<unsigned int> indices(indexBufferSize);
    indexFile.read(reinterpret_cast<char*>(indices.data()), indices.size() * sizeof(unsigned int));
    indexFile.close();

    mesh = Mesh({
        .vertices = vertices,
        .indices = indices,
        .material = new UnlitMaterial({ .diffuseTextureID = colorTexture.ID }),
        .wireframe = false,
        .pointcloud = renderState == RenderState::POINTCLOUD,
    });

    wireframeMesh = Mesh({
        .vertices = vertices,
        .indices = indices,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .wireframe = true,
        .pointcloud = false,
    });

    return vertices.size();
}

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp2 Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'i', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
    args::ValueFlag<int> renderStateIn(parser, "render", "Render state", {'r', "render-state"}, 0);
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find("x");
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    renderState = static_cast<RenderState>(args::get(renderStateIn));
    surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

    int trianglesDrawn = 0;
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
        ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

        ImGui::Separator();

        if (trianglesDrawn < 100000) {
            ImGui::TextColored(ImVec4(0,1,0,1), "Total Triangles Drawn: %d", trianglesDrawn);
        }
        else if (trianglesDrawn < 500000) {
            ImGui::TextColored(ImVec4(1,1,0,1), "Total Triangles Drawn: %d", trianglesDrawn);
        }
        else {
            ImGui::TextColored(ImVec4(1,0,0,1), "Total Triangles Drawn: %d", trianglesDrawn);
        }

        ImGui::Separator();

        glm::vec3 position = camera.getPosition();
        ImGui::InputFloat3("Camera Position", (float*)&position);
        camera.setPosition(position);
        ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 0.1f, 20.0f);

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
    std::vector<Mesh> wireframeMeshes(labels.size());
    std::vector<Node> nodes(2*labels.size());
    for (int i = 0; i < labels.size(); i++) {
        createMesh(meshes[i], wireframeMeshes[i], labels[i]);

        nodes[2*i] = Node(&meshes[i]);
        scene.addChildNode(&nodes[i]);

        nodes[2*i+1] = Node(&wireframeMeshes[i]);
        scene.addChildNode(&nodes[i+1]);
    }

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    // load camera view and projection matrices
    std::ifstream cameraFile(DATA_PATH + "data/camera.bin", std::ios::binary);
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
        }

        // handle keyboard input
        auto keys = window->getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window->close();
        }

        for (auto& mesh : meshes) {
            mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        }
        for (auto& mesh : wireframeMeshes) {
            mesh.visible = renderState == RenderState::WIREFRAME;
        }

        nodes[1].setPosition(nodes[0].getPosition() - camera.getForwardVector() * 0.0005f);

        // render all objects in scene
        trianglesDrawn = app.renderer->drawObjects(scene, camera);

        // render to screen
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
