#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowDepthEffect.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Streamers/BC4DepthStreamer.h>
#include <Streamers/PoseStreamer.h>

#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 32

using namespace quasar;

enum class RenderState {
    MESH,
    POINTCLOUD
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "BC4 Compression";

    RenderState renderState = RenderState::POINTCLOUD;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<uint> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
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

    if (verbose) spdlog::set_level(spdlog::level::debug);

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);

    Path sceneFile = args::get(sceneFileIn);

    uint surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer remoteRenderer(config);
    ForwardRenderer renderer(config);

    // "Remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(windowSize);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    // Scene with all the meshes
    Scene scene = Scene();
    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    PerspectiveCamera camera = PerspectiveCamera(windowSize);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    BC4DepthStreamer bc4DepthStreamerRT = BC4DepthStreamer({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    });

    // Shaders
    ComputeShader meshFromDepthShader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_DEPTH_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_DEPTH_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    ComputeShader meshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_BC4_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_BC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    // Post processing
    ToneMapper toneMapper;
    ShowDepthEffect showDepthEffect(camera);

    // Original size of depth buffer
    uint originalSize = windowSize.x * windowSize.y * sizeof(float);

    // Create buffer for compressed data
    uint compressedSize = (windowSize.x / 8) * (windowSize.y / 8) * sizeof(BC4Block);
    float compressionRatio = originalSize / compressedSize;

    // Set up meshes for rendering
    glm::uvec2 adjustedWindowSize = windowSize / surfelSize;

    uint maxVertices = adjustedWindowSize.x * adjustedWindowSize.y;
    uint numTriangles = (adjustedWindowSize.x-1) * (adjustedWindowSize.y-1) * 2;
    uint maxIndices = numTriangles * 3;

    Mesh mesh = Mesh({
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    node.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    node.pointSize = 7.5f;
    scene.addChildNode(&node);

    Mesh meshDecompressed = Mesh({
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDecompressed = Node(&meshDecompressed);
    nodeDecompressed.frustumCulled = false;
    nodeDecompressed.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    nodeDecompressed.pointSize = 7.5f;
    scene.addChildNode(&nodeDecompressed);

    bool sendRemoteFrame = true;
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showFrameCaptureWindow = false;
        static bool showDepthPreview = true;

        ImGui::NewFrame();

        ImGuiWindowFlags flags = 0;
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
            ImGui::MenuItem("Frame Capture", 0, &showFrameCaptureWindow);
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
            if (ImGui::DragFloat3("Camera Position", (float*)&position, 0.01f)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::DragFloat3("Camera Rotation", (float*)&rotation, 0.1f)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::DragFloat("Movement Speed", &camera.movementSpeed, 0.05f, 0.1f, 20.0f);

            ImGui::Separator();

            ImGui::RadioButton("Display Mesh", (int*)&renderState, 0);
            ImGui::RadioButton("Display Point Cloud", (int*)&renderState, 1);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0,1,1), "Original Depth Buffer");
            ImGui::TextColored(ImVec4(1,1,0,1), "Decompressed Depth Buffer");

            ImGui::Separator();

            ImGui::Checkbox("Show Original Depth", &node.visible);
            ImGui::Checkbox("Show Decompressed Depth", &nodeDecompressed.visible);

            ImGui::Separator();

            if (ImGui::Button("Send Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                sendRemoteFrame = true;
            }

            ImGui::End();
        }

        if (showDepthPreview) {
            flags = ImGuiWindowFlags_AlwaysAutoResize;
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - windowSize.x / 4 - 60, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Original Depth", &showDepthPreview, flags);
            ImGui::Image((void*)(intptr_t)remoteRenderer.frameRT.depthStencilTexture, ImVec2(windowSize.x / 4, windowSize.y / 4), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }
    });

    // Window resize callback
    app.onResize([&](uint width, uint height) {
        windowSize.x = width;
        windowSize.y = height;

        remoteRenderer.setWindowSize(width, height);
        renderer.setWindowSize(width, height);
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });

    app.onRender([&](double now, double dt) {
        // Handle mouse input
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
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }
        auto scroll = window->getScrollOffset();
        camera.processScroll(scroll.y);
        camera.processKeyboard(keys, dt);

        if (sendRemoteFrame) {
            remoteCamera.setPosition(camera.getPosition());
            remoteCamera.setRotationQuat(camera.getRotationQuat());
            remoteCamera.updateViewMatrix();

            remoteRenderer.drawObjects(remoteScene, remoteCamera);

            sendRemoteFrame = false;
        }

        // Generate mesh for original depth data
        meshFromDepthShader.bind();
        {
            meshFromDepthShader.setTexture(remoteRenderer.frameRT.depthStencilTexture, 0);
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
        // Dispatch compute shader to generate vertices for mesh
        meshFromDepthShader.dispatch((adjustedWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                     (adjustedWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        meshFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                          GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // Compress with BC4
        showDepthEffect.drawToRenderTarget(remoteRenderer, bc4DepthStreamerRT);
        bc4DepthStreamerRT.compress();

        // Generate mesh using compressed depth data
        meshFromBC4Shader.bind();
        {
            meshFromBC4Shader.setBool("unlinearizeDepth", true);
            meshFromBC4Shader.setVec2("depthMapSize", windowSize);
            meshFromBC4Shader.setInt("surfelSize", surfelSize);
        }
        {
            meshFromBC4Shader.setMat4("projection", remoteCamera.getProjectionMatrix());
            meshFromBC4Shader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
            meshFromBC4Shader.setMat4("viewColor", remoteCamera.getViewMatrix());
            meshFromBC4Shader.setMat4("viewInverseDepth", remoteCamera.getViewMatrixInverse());

            meshFromBC4Shader.setFloat("near", remoteCamera.getNear());
            meshFromBC4Shader.setFloat("far", remoteCamera.getFar());
        }
        {
            meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshDecompressed.vertexBuffer);
            meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, meshDecompressed.indexBuffer);
            meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, bc4DepthStreamerRT.bc4CompressedBuffer);
        }
        // Dispatch compute shader to generate vertices and indices for mesh
        meshFromBC4Shader.dispatch((adjustedWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                   (adjustedWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // Set render state
        node.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
        nodeDecompressed.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        toneMapper.enableToneMapping(false);
        toneMapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
