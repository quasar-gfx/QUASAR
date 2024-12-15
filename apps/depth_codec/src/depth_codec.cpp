#include <iostream>
#include <iomanip>
#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <Shaders/ToneMapShader.h>

#include <Recorder.h>
#include <Animator.h>

#include <VideoTexture.h>
#include <DepthVideoTexture.h>
#include <PoseStreamer.h>
#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16

#define TEXTURE_PREVIEW_SIZE 250

const std::string DATA_PATH = "./";

enum class RenderState {
    MESH,
    POINTCLOUD
};

struct Block {
    float max; // 32 - unit32
    float min;
    uint32_t arr[6];
    //float real[64];
};

// Function to calculate MSE
double calculateMSE(const std::vector<float>& original, const std::vector<float>& decompressed) {
    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double diff = original[i] - decompressed[i];
        mse += diff * diff;
    }
    return mse / original.size();
}

int main(int argc, char** argv) {
    Config config{};
    config.title = "BC4 Compression";

    RenderState renderState = RenderState::POINTCLOUD;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<unsigned int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
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

    std::string sceneFile = args::get(sceneFileIn);

    unsigned int surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer remoteRenderer(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    // "remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    // scene with all the meshes
    Scene scene = Scene();
    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    PerspectiveCamera camera = PerspectiveCamera(windowSize.x, windowSize.y);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    // shaders
    ToneMapShader toneMapShader;

    Shader videoShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_DISPLAYTEXTURE_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_DISPLAYTEXTURE_FRAG_len
    });

    ComputeShader meshFromDepthShader({
        .computeCodeData = SHADER_COMMON_MESHFROMDEPTH_COMP,
        .computeCodeSize = SHADER_COMMON_MESHFROMDEPTH_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader bc4CompressionShader({
        .computeCodeData = SHADER_COMMON_BC4COMPRESSION_COMP,
        .computeCodeSize = SHADER_COMMON_BC4COMPRESSION_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader genMeshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_GENMESHFROMBC4_COMP,
        .computeCodeSize = SHADER_COMMON_GENMESHFROMBC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    // original size of depth buffer
    unsigned int originalSize = windowSize.x * windowSize.y * sizeof(float);

    // create buffer for compressed data
    unsigned int compressedSize = (windowSize.x / 8) * (windowSize.y / 8) * sizeof(Block);
    Buffer<Block> bc4Buffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, compressedSize / sizeof(Block), nullptr);

    float compressionRatio = originalSize / compressedSize;

    // set up meshes for rendering
    glm::uvec2 adjustedWindowSize = windowSize / surfelSize;

    int numVertices = adjustedWindowSize.x * adjustedWindowSize.y;

    int numTriangles = (adjustedWindowSize.x-1) * (adjustedWindowSize.y-1) * 2;
    int indexBufferSize = numTriangles * 3;

    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    node.primativeType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    node.pointSize = 7.5f;
    scene.addChildNode(&node);

    Mesh meshDecompressed = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDecompressed = Node(&meshDecompressed);
    nodeDecompressed.frustumCulled = false;
    nodeDecompressed.primativeType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    nodeDecompressed.pointSize = 7.5f;
    scene.addChildNode(&nodeDecompressed);

    bool rerender = true;
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool showDepthPreview = true;

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

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Compression Ratio: %d:1", static_cast<int>(compressionRatio));

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

            ImGui::Separator();

            if (ImGui::Button("Send Server Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                rerender = true;
            }

            ImGui::End();
        }

        if (showDepthPreview) {
            flags = ImGuiWindowFlags_AlwaysAutoResize;
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - TEXTURE_PREVIEW_SIZE - 30, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Original Depth", &showDepthPreview, flags);
            ImGui::Image((void*)(intptr_t)remoteRenderer.gBuffer.depthStencilBuffer, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }
    });

    // Window resize callback
    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize.x = width;
        windowSize.y = height;

        remoteRenderer.setWindowSize(width, height);
        renderer.setWindowSize(width, height);

        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    app.onRender([&](double now, double dt) {
        // handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = windowSize.x / 2.0;
            static float lastY = windowSize.y / 2.0;
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
            remoteCamera.setPosition(camera.getPosition());
            remoteCamera.setRotationQuat(camera.getRotationQuat());
            remoteCamera.updateViewMatrix();

            remoteRenderer.drawObjects(remoteScene, remoteCamera);

            rerender = false;
        }

        // generate mesh for original depth data
        meshFromDepthShader.bind();
        {
            meshFromDepthShader.setTexture(remoteRenderer.gBuffer.depthStencilBuffer, 0);
        }
        {
            meshFromDepthShader.setVec2("depthMapSize", windowSize);
            meshFromDepthShader.setInt("surfelSize", surfelSize);
        }
        {
            meshFromDepthShader.setMat4("projection", remoteCamera.getProjectionMatrix());
            meshFromDepthShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
            meshFromDepthShader.setMat4("view", remoteCamera.getViewMatrix());
            meshFromDepthShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());

            meshFromDepthShader.setFloat("near", remoteCamera.getNear());
            meshFromDepthShader.setFloat("far", remoteCamera.getFar());
        }
        {
            meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
            meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
        }
        // dispatch compute shader to generate vertices for mesh
        genMeshFromBC4Shader.dispatch((adjustedWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                      (adjustedWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        meshFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                     GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // compress depth data using BC4
        bc4CompressionShader.bind();
        {
            bc4CompressionShader.setTexture(remoteRenderer.gBuffer.depthStencilBuffer, 0);
        }
        {
            bc4CompressionShader.setVec2("depthMapSize", windowSize);
            bc4CompressionShader.setVec2("bc4DepthSize", glm::vec2(windowSize.x / 8, windowSize.y / 8));
        }
        {
            bc4CompressionShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, bc4Buffer);
        }
        bc4CompressionShader.dispatch((windowSize.x / 8) / THREADS_PER_LOCALGROUP, (windowSize.y / 8) / THREADS_PER_LOCALGROUP, 1);
        bc4CompressionShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // generate mesh using compressed depth data
        genMeshFromBC4Shader.bind();
        {
            genMeshFromBC4Shader.setVec2("depthMapSize", windowSize);
            genMeshFromBC4Shader.setInt("surfelSize", surfelSize);

            genMeshFromBC4Shader.setBool("unlinearizeDepth", false);
        }
        {
            genMeshFromBC4Shader.setMat4("projection", remoteCamera.getProjectionMatrix());
            genMeshFromBC4Shader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
            genMeshFromBC4Shader.setMat4("viewColor", remoteCamera.getViewMatrix());
            genMeshFromBC4Shader.setMat4("viewInverseDepth", remoteCamera.getViewMatrixInverse());

            genMeshFromBC4Shader.setFloat("near", remoteCamera.getNear());
            genMeshFromBC4Shader.setFloat("far", remoteCamera.getFar());
        }
        {
            genMeshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshDecompressed.vertexBuffer);
            genMeshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, meshDecompressed.indexBuffer);
            genMeshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, bc4Buffer);

        }
        // dispatch compute shader to generate vertices and indices for mesh
        genMeshFromBC4Shader.dispatch((adjustedWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                      (adjustedWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        genMeshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                           GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // set render state
        node.primativeType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
        nodeDecompressed.primativeType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        toneMapShader.bind();
        toneMapShader.setBool("toneMap", false);
        renderer.drawToScreen(toneMapShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
