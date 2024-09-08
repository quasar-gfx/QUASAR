#include <iostream>

#include <args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <VideoTexture.h>
#include <DepthVideoTexture.h>
#include <PoseStreamer.h>

#define TEXTURE_PREVIEW_SIZE 250

#define VERTICES_IN_A_QUAD 4

const std::string DATA_PATH = "./";

enum class RenderState {
    MESH,
    POINTCLOUD
};

std::vector<uint16_t> convert32To16Bit(const std::vector<float> &image) {
    std::vector<uint16_t> convertedImage(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        convertedImage[i] = static_cast<uint16_t>(image[i] * 65535.0f);
    }
    return convertedImage;
}

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Reciever";

    RenderState renderState = RenderState::POINTCLOUD;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
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
    size_t pos = sizeStr.find('x');
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    int surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene scene = Scene();
    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    PerspectiveCamera camera = PerspectiveCamera(screenWidth, screenHeight);
    PerspectiveCamera origCamera = PerspectiveCamera(screenWidth, screenHeight);

    camera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    camera.updateViewMatrix();
    origCamera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    origCamera.updateViewMatrix();

    std::vector<float> depthData(screenWidth * screenHeight);
    auto depthFile = FileIO::loadBinaryFile(DATA_PATH + "depth.bin");
    std::memcpy(depthData.data(), depthFile.data(), depthFile.size());

    Texture depthTextureOriginal({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_R16F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .data = reinterpret_cast<unsigned char*>(depthData.data())
    });

    std::vector<float> depthDataDecompressed(screenWidth * screenHeight);
    // fill with random values for now
    for (size_t i = 0; i < depthDataDecompressed.size(); ++i) {
        depthDataDecompressed[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    Texture depthTextureDecompressed({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_R16F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .data = reinterpret_cast<unsigned char*>(depthDataDecompressed.data())
    });

    // shaders
    Shader screenShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader videoShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayTexture.frag",
    });

    ComputeShader genPtCloudFromDepthShader({
        .computeCodePath = "./shaders/genPtCloudFromDepth.comp"
    });

    int width = screenWidth / surfelSize;
    int height = screenHeight / surfelSize;

    int numVertices = width * height;

    int numTriangles = (width-1) * (height-1) * 2;
    int indexBufferSize = numTriangles * 3;

    genPtCloudFromDepthShader.bind();
    genPtCloudFromDepthShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genPtCloudFromDepthShader.setInt("surfelSize", surfelSize);
    genPtCloudFromDepthShader.unbind();

    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) }),
        .pointcloud = renderState == RenderState::POINTCLOUD,
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Mesh meshDecompressed = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .pointcloud = renderState == RenderState::POINTCLOUD,
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDecompressed = Node(&meshDecompressed);
    nodeDecompressed.frustumCulled = false;
    scene.addChildNode(&nodeDecompressed);

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool showDepthPreview = true;

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);

        ImGui::NewFrame();

        unsigned int flags = 0;
        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &showFPS);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Frame Capture", 0, &showCaptureWindow);
            ImGui::MenuItem("Depth Preview", 0, &showDepthPreview);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        if (showFPS) {
            ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
            flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
            ImGui::Begin("", 0, flags);
            ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
            ImGui::End();
        }

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Draw Calls: %d", renderStats.drawCalls);

            ImGui::Separator();

            glm::vec3 position = camera.getPosition();
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                camera.setRotationEuler(rotation);
            }

            ImGui::Separator();

            ImGui::RadioButton("Display Mesh", (int*)&renderState, 0);
            ImGui::RadioButton("Display Point Cloud", (int*)&renderState, 1);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0,0,1), "Original Depth Buffer");
            ImGui::TextColored(ImVec4(1,1,0,1), "Decompressed Depth Buffer");

            ImGui::Separator();

            ImGui::Checkbox("Show Original Depth", &node.visible);
            ImGui::Checkbox("Show Decompressed Depth", &nodeDecompressed.visible);

            ImGui::End();
        }

        flags = ImGuiWindowFlags_AlwaysAutoResize;

        if (showDepthPreview) {
            ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Original Depth", &showDepthPreview, flags);
            ImGui::Image((void*)(intptr_t)depthTextureOriginal.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 70 + TEXTURE_PREVIEW_SIZE + 30), ImGuiCond_FirstUseEver);
            ImGui::Begin("Decompressed Depth", &showDepthPreview, flags);
            ImGui::Image((void*)(intptr_t)depthTextureDecompressed.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        renderer.resize(width, height);

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

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

        // generate mesh for original and decompressed depth data
        genPtCloudFromDepthShader.bind();
        {
            genPtCloudFromDepthShader.setMat4("view", origCamera.getViewMatrix());
            genPtCloudFromDepthShader.setMat4("projection", origCamera.getProjectionMatrix());
            genPtCloudFromDepthShader.setMat4("viewInverse", glm::inverse(origCamera.getViewMatrix()));
            genPtCloudFromDepthShader.setMat4("projectionInverse", glm::inverse(origCamera.getProjectionMatrix()));
        }
        {
            genPtCloudFromDepthShader.setTexture(depthTextureOriginal, 0);
        }
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
        }
        // dispatch compute shader to generate vertices and indices for mesh
        genPtCloudFromDepthShader.dispatch(width / 16, height / 16, 1);
        genPtCloudFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // do it again with decompressed depth data:
        {
            genPtCloudFromDepthShader.setTexture(depthTextureDecompressed, 0);
        }
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meshDecompressed.vertexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshDecompressed.indexBuffer);
        }
        // dispatch compute shader to generate vertices and indices for mesh
        genPtCloudFromDepthShader.dispatch(width / 16, height / 16, 1);
        genPtCloudFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        genPtCloudFromDepthShader.unbind();

        // set render state
        mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        meshDecompressed.pointcloud = renderState == RenderState::POINTCLOUD;

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        screenShader.bind();
        screenShader.setBool("doToneMapping", false);
        renderer.drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
