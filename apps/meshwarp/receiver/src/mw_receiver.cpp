#include <args/args.hxx>
#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <PostProcessing/ToneMapper.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>
#include <VideoTexture.h>
#include <BC4DepthVideoTexture.h>
#include <PoseStreamer.h>
#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16

using namespace quasar;

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<uint> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "0.0.0.0:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "127.0.0.1:54321");
    args::ValueFlag<uint> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<float> fovIn(parser, "fov", "Field of view", {'f', "fov"}, 60.0f);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
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

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    }

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);

    std::string sceneFile = args::get(sceneFileIn);
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();

    uint surfelSize = args::get(surfelSizeIn);
    uint depthFactor = args::get(depthFactorIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    RenderState renderState = RenderState::MESH;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    VideoTexture videoTextureColor({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoURL);

    BC4DepthVideoTexture videoTextureDepth({
        .width = windowSize.x / depthFactor,
        .height = windowSize.y / depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthURL);

    // "Remote" camera
    PerspectiveCamera remoteCamera(videoTextureColor.width, videoTextureColor.height);
    remoteCamera.setFovyDegrees(args::get(fovIn));

    // "Local" scene
    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);

    PoseStreamer poseStreamer(&camera, poseURL);

    glm::uvec2 adjustedWindowSize = windowSize / surfelSize;

    uint maxVertices = adjustedWindowSize.x * adjustedWindowSize.y;
    uint numTriangles = (adjustedWindowSize.x-1) * (adjustedWindowSize.y-1) * 2;
    uint maxIndices = numTriangles * 3;

    Mesh mesh = Mesh({
        .maxVertices = maxVertices,
        .maxIndices = maxIndices,
        .material = new UnlitMaterial({ .baseColorTexture = &videoTextureColor ,}),
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    node.primativeType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
    scene.addChildNode(&node);

    Node nodeWireframe = Node(&mesh);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    nodeWireframe.overrideMaterial = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) });
    scene.addChildNode(&nodeWireframe);

    // Post processing
    ToneMapper toneMapper;
    toneMapper.enableToneMapping(false);

    Shader videoShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_SHOW_TEXTURE_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_SHOW_TEXTURE_FRAG_len,
    });

    ComputeShader meshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_BC4_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_BC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    });

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, toneMapper, outputPath, config.targetFramerate);

    double elapsedTimeColor, elapsedTimeDepth;
    pose_id_t poseIdColor = -1, poseIdDepth = -1;
    bool mwEnabled = true;
    bool sync = true;
    Pose currentColorFramePose, currentDepthFramePose;

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showFrameCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showVideoPreview = true;

        ImGui::NewFrame();

        uint flags = 0;
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
            ImGui::MenuItem("Video Preview", 0, &showVideoPreview);
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
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %d", renderStats.drawCalls);

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

            ImGui::Text("Remote Pose ID: RGB (%d), D (%d)", poseIdColor, poseIdDepth);

            glm::mat4 pose = glm::inverse(currentDepthFramePose.mono.view);
            glm::vec3 skew, scale;
            glm::quat rotationQuat;
            glm::vec3 remotePosition;
            glm::vec4 perspective;
            glm::decompose(pose, scale, rotationQuat, remotePosition, skew, perspective);
            glm::vec3 remoteRotation = glm::degrees(glm::eulerAngles(rotationQuat));
            ImGui::BeginDisabled();
            ImGui::InputFloat3("Remote Position", (float*)&remotePosition);
            ImGui::InputFloat3("Remote Rotation", (float*)&remoteRotation);
            ImGui::EndDisabled();

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Depth URL: %s", depthURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f FPS), D (%.1f FPS)", videoTextureColor.getFrameRate(), videoTextureDepth.getFrameRate());
            ImGui::TextColored(ImVec4(1,0.5,0,1), "E2E Latency: RGB (%.3f ms), D (%.3f ms)", elapsedTimeColor, elapsedTimeDepth);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms", videoTextureColor.stats.timeToReceiveMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: RGB (%.3f Mbps), D (%.3f Mbps)", videoTextureColor.stats.bitrateMbps, videoTextureDepth.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::Checkbox("Mesh Warp Enabled", &mwEnabled);

            ImGui::Separator();

            ImGui::Checkbox("Sync Color and Depth", &sync);

            ImGui::Separator();

            ImGui::RadioButton("Show Mesh", (int*)&renderState, 0);
            ImGui::RadioButton("Show Point Cloud", (int*)&renderState, 1);
            ImGui::RadioButton("Show Wireframe", (int*)&renderState, 2);

            ImGui::End();
        }

        flags = ImGuiWindowFlags_AlwaysAutoResize;
        if (showVideoPreview) {
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - windowSize.x / 4 - 60, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Raw Video Texture", &showVideoPreview, flags);
            ImGui::Image((void*)(intptr_t)videoTextureColor.ID, ImVec2(windowSize.x / 4, windowSize.y / 4), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        if (showFrameCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showFrameCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string time = std::to_string(static_cast<int>(window->getTime() * 1000.0f));
            Path filename = (outputPath / fileNameBase).appendToName("." + time);

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(filename, saveAsHDR);
            }

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);

        camera.setAspect(windowSize.x, windowSize.y);
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
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window->close();
        }
        auto scroll = window->getScrollOffset();
        camera.processScroll(scroll.y);

        // Send pose to streamer
        poseStreamer.sendPose();

        // Render color video frame
        videoTextureColor.bind();
        poseIdColor = videoTextureColor.draw();
        videoTextureColor.unbind();

        // Render depth video frame
        if (sync) {
            poseIdDepth = videoTextureDepth.draw(poseIdColor);
        }
        else {
            poseIdDepth = videoTextureDepth.draw();
        }

        if (!mwEnabled) {
            if (poseIdColor != -1) poseStreamer.getPose(poseIdColor, &currentColorFramePose, &elapsedTimeColor);

            videoShader.bind();
            videoShader.setTexture("tex", videoTextureColor, 5);
            renderStats = renderer.drawToScreen(videoShader);

            return;
        }

        // Set shader uniforms
        meshFromBC4Shader.bind();
        {
            meshFromBC4Shader.setBool("unlinearizeDepth", true);
            meshFromBC4Shader.setVec2("depthMapSize", glm::vec2(videoTextureDepth.width, videoTextureDepth.height));
            meshFromBC4Shader.setInt("surfelSize", surfelSize);
        }
        {
            meshFromBC4Shader.setMat4("projection", remoteCamera.getProjectionMatrix());
            meshFromBC4Shader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
            if (poseStreamer.getPose(poseIdColor, &currentColorFramePose, &elapsedTimeColor)) {
                meshFromBC4Shader.setMat4("viewColor", currentColorFramePose.mono.view);
            }
            if (poseStreamer.getPose(poseIdDepth, &currentDepthFramePose, &elapsedTimeDepth)) {
                meshFromBC4Shader.setMat4("viewInverseDepth", glm::inverse(currentDepthFramePose.mono.view));
            }

            meshFromBC4Shader.setFloat("near", remoteCamera.getNear());
            meshFromBC4Shader.setFloat("far", remoteCamera.getFar());
        }
        {
            meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
            meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
            meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, videoTextureDepth.bc4CompressedBuffer);
        }

        // Dispatch compute shader to generate vertices and indices for both main and wireframe meshes
        meshFromBC4Shader.dispatch(
                ((videoTextureDepth.width / surfelSize)  + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                ((videoTextureDepth.height / surfelSize) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                1
            );
        meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        poseStreamer.removePosesLessThan(std::min(poseIdColor, poseIdDepth));

        // Set render state
        node.primativeType = renderState == RenderState::POINTCLOUD ? GL_POINTS : GL_TRIANGLES;
        nodeWireframe.visible = renderState == RenderState::WIREFRAME;

        // Render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // Render to screen
        toneMapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
