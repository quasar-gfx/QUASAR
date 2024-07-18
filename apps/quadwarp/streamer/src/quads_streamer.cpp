#include <iostream>

#include <args.hxx>
#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Materials/UnlitMaterial.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <RenderTargets/RenderTarget.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <SceneLoader.h>

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

const std::string DATA_PATH = "./";

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int surfelSize = 4;
RenderState renderState = RenderState::MESH;

int main(int argc, char** argv) {
    Config config{};
    config.title = "QuadWarp Streamer";
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> size2In(parser, "size", "Size of pre-rendered content", {'S', "size2"}, "800x600");
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

    // parse size2
    std::string size2Str = args::get(size2In);
    pos = size2Str.find("x");
    int size2Width = std::stoi(size2Str.substr(0, pos));
    int size2Height = std::stoi(size2Str.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    int surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Scene remoteScene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);
    Camera remoteCamera = Camera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, remoteScene, remoteCamera);

    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    int trianglesDrawn = 0;
    bool rerender = true;
    int showNormals = 0;
    bool showNormalsChanged = false;
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

        ImGui::Separator();

        int prevshowNormals = showNormals;
        ImGui::RadioButton("Show Color", (int*)&showNormals, 0);
        ImGui::RadioButton("Show Normals", (int*)&showNormals, 1);
        if (prevshowNormals != showNormals) {
            showNormalsChanged = true;
            rerender = true;
        }

        ImGui::Separator();

        if (ImGui::Button("Rerender", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            rerender = true;
        }
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

    Shader screenShader2Color = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader screenShader2Normals = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayNormals.frag"
    });

    ComputeShader genMeshShader({
        .computeCodePath = "./shaders/genQuads.comp"
    });

    unsigned int remoteWidth = size2Width / surfelSize;
    unsigned int remoteHeight = size2Height / surfelSize;

    RenderTarget renderTarget({
        .width = remoteWidth,
        .height = remoteHeight,
        .internalFormat = GL_RGBA16,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    GLuint vertexBuffer;
    int numVertices = remoteWidth * remoteHeight * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(Vertex), nullptr, GL_STATIC_DRAW);

    GLuint indexBuffer;
    int numTriangles = remoteWidth * remoteHeight * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    genMeshShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    genMeshShader.setVec2("screenSize", glm::vec2(remoteWidth, remoteHeight));
    genMeshShader.setInt("surfelSize", surfelSize);
    genMeshShader.unbind();

    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .diffuseTextureID = renderTarget.colorBuffer.ID }),
        .wireframe = false,
        .pointcloud = renderState == RenderState::POINTCLOUD,
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Mesh meshWireframe = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .wireframe = true,
        .pointcloud = false,
    });
    Node nodeWireframe = Node(&meshWireframe);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.setPosition(glm::vec3(0.0f, 0.001f, 0.001f));
    scene.addChildNode(&nodeWireframe);

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

        if (rerender) {
            if (!showNormalsChanged) {
                remoteCamera.setPosition(camera.getPosition());
                remoteCamera.setRotationQuat(camera.getRotationQuat());
                remoteCamera.updateViewMatrix();
            }
            showNormalsChanged = false;

            double startTime = glfwGetTime();

            // render all objects in remoteScene
            app.renderer->drawObjects(remoteScene, remoteCamera);

            // render to render target
            if (showNormals == 0) {
                app.renderer->drawToRenderTarget(screenShader2Color, renderTarget);
            }
            else {
                app.renderer->drawToRenderTarget(screenShader2Normals, renderTarget);
            }

            genMeshShader.bind();
            genMeshShader.setMat4("view", remoteCamera.getViewMatrix());
            genMeshShader.setMat4("projection", remoteCamera.getProjectionMatrix());
            genMeshShader.setMat4("viewInverse", glm::inverse(remoteCamera.getViewMatrix()));
            genMeshShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
            genMeshShader.setFloat("near", remoteCamera.near);
            genMeshShader.setFloat("far", remoteCamera.far);
            app.renderer->gBuffer.positionBuffer.bind(0);
            app.renderer->gBuffer.normalsBuffer.bind(1);
            app.renderer->gBuffer.idBuffer.bind(2);
            app.renderer->gBuffer.depthBuffer.bind(3);
            genMeshShader.dispatch(remoteWidth, remoteHeight, 1);
            genMeshShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            genMeshShader.unbind();

            std::cout << "Rendering Time: " << glfwGetTime() - startTime << "s" << std::endl;

            mesh.setBuffers(vertexBuffer, indexBuffer);
            meshWireframe.setBuffers(vertexBuffer, indexBuffer);

            rerender = false;
        }

        mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        meshWireframe.visible = renderState == RenderState::WIREFRAME;

        nodeWireframe.setPosition(node.getPosition() - camera.getForwardVector() * 0.0005f);

        // render all objects in scene
        trianglesDrawn = app.renderer->drawObjects(scene, camera);

        // render to screen
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
