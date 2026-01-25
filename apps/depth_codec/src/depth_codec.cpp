#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowDepthEffect.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/ScreenshotWindow.h>
#include <UI/RecordWindow.h>
#include <UI/TexturePreviewWindow.h>
#include <UI/SceneWindow.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <Streamers/BC4DepthStreamer.h>
#include <Streamers/PoseStreamer.h>

#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

enum class RenderState {
    MESH,
    POINTCLOUD
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "Depth Compression";

    RenderState renderState = RenderState::POINTCLOUD;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> verbosity(parser, "verbosity", "Set log verbosity level", {'v', "verbosity"}, 2 /* spdlog::level::info */);
    args::ValueFlag<std::string> sizeIn(parser, "size", "Window resolution", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<int> numPosesIn(parser, "num-poses", "Number of poses to load from camera path", {'n', "num-poses"}, -1);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<uint> vertexGroupSizeIn(parser, "vertex", "Size of vertex grouping", {'g', "vertex-group-size"}, 1);
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



    // Parse arguments
    bool saveImagesToDisk = args::get(saveImages);
    int numPoses = args::get(numPosesIn);
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);
    uint vertexGroupSize = args::get(vertexGroupSizeIn);

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.verbosity = args::get(verbosity);
    config.enableVSync = !args::get(novsync) && !saveImagesToDisk;
    config.showWindow = !saveImagesToDisk;

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

    BC4DepthStreamer bc4DepthStreamerRT({
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
    Tonemapper tonemapper;
    ShowDepthEffect showDepthEffect(camera);

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, tonemapper, outputPath, config.targetFramerate);

    // Original size of depth buffer
    uint originalSize = windowSize.x * windowSize.y * sizeof(float);

    // Create buffer for compressed data
    uint compressedSize = (windowSize.x / 8) * (windowSize.y / 8) * sizeof(BC4Block);
    float compressionRatio = originalSize / compressedSize;

    // Set up meshes for rendering
    glm::uvec2 adjustedWindowSize = windowSize / vertexGroupSize;

    uint maxVertices = adjustedWindowSize.x * adjustedWindowSize.y;
    uint numTriangles = (adjustedWindowSize.x-1) * (adjustedWindowSize.y-1) * 2;
    uint maxIndices = numTriangles * 3;

    UnlitMaterial meshMaterial({ .baseColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) });
    Mesh mesh({
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .material = &meshMaterial,
        .usage = GL_DYNAMIC_DRAW
    });
    Node node(&mesh);
    node.frustumCulled = false;
    node.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    node.pointSize = 7.5f;
    scene.addChildNode(&node);

    UnlitMaterial meshDecompressedMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    Mesh meshDecompressed({
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .material = &meshDecompressedMaterial,
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDecompressed(&meshDecompressed);
    nodeDecompressed.frustumCulled = false;
    nodeDecompressed.primitiveType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    nodeDecompressed.pointSize = 7.5f;
    scene.addChildNode(&nodeDecompressed);

    bool sendRemoteFrame = true;

    FrameRateWindow frameRateWindow;
    ScreenshotWindow screenshotWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    TexturePreviewWindow depthPreviewWindow("Depth", remoteRenderer.frameRT.depthStencilTexture, ImVec2(430, 270));
    SceneWindow sceneWindow(scene, ImVec2(430, 800));
    CameraHeader cameraHeader(camera);

    CameraAnimator cameraAnimator(cameraPathFile, numPoses, !saveImagesToDisk); // Disable tweening when saving images
    if (saveImagesToDisk || cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        spdlog::info("Loading camera path {} and saving images to {}", cameraPathFile.str(), outputPath.str());

        if (saveImagesToDisk) {
            recorder.setTargetFrameRate(-1 /* unlimited */);
            recorder.setFormat(Recorder::OutputFormat::PNG);
            recorder.start();
        }
    }

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImagesToDisk;

        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &frameRateWindow.visible);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Depth Preview", 0, &depthPreviewWindow.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Scene", 0, &sceneWindow.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Frame Capture")) {
            ImGui::MenuItem("Take Screenshot", 0, &screenshotWindow.visible);
            ImGui::MenuItem("Record Video", 0, &recordWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        screenshotWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        depthPreviewWindow.draw(now, dt);
        sceneWindow.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Triangles Drawn: %ld", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Draw Calls: %ld", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Compression Ratio: %d:1", static_cast<int>(compressionRatio));

            ImGui::Separator();

            cameraHeader.draw(now, dt);

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

        if (cameraAnimator.isRunning()) {
            bool waypointUpdated = cameraAnimator.update(saveImagesToDisk ? (1.0 / 60.0) : dt);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (waypointUpdated) {
                cameraAnimator.copyPoseToCamera(camera);
            }
        }
        else {
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
            camera.processKeyboard(keys, dt);
        }

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
            meshFromDepthShader.setUint("vertexGroupSize", vertexGroupSize);
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
        bc4DepthStreamerRT.generateFrame();

        // Generate mesh using compressed depth data
        meshFromBC4Shader.bind();
        {
            meshFromBC4Shader.setBool("unlinearizeDepth", true);
            meshFromBC4Shader.setVec2("depthMapSize", windowSize);
            meshFromBC4Shader.setUint("vertexGroupSize", vertexGroupSize);
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
        tonemapper.enableTonemapping(false);
        tonemapper.drawToScreen(renderer);

        if (saveImagesToDisk) {
            recorder.captureFrame(camera);

            if (!cameraAnimator.isRunning()) {
                recorder.stop();
                window->close();
            }
        }
        else if (recordWindow.isRecording()) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
