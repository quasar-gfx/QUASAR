#include <iostream>

#include <args/args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <VideoTexture.h>
#include <DepthVideoTexture.h>
#include <PoseStreamer.h>

#define TEXTURE_PREVIEW_SIZE 500

#define VERTICES_IN_A_QUAD 4

const std::string DATA_PATH = "../streamer/";

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Reciever";

    RenderState renderState = RenderState::MESH;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "0.0.0.0:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "127.0.0.1:54321");
    args::ValueFlag<int> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
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
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    int surfelSize = args::get(surfelSizeIn);
    int depthFactor = args::get(depthFactorIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    glm::uvec2 windowSize = window->getSize();

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);

    VideoTexture videoTextureColor({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL);
    DepthVideoTexture videoTextureDepth({
        .width = windowSize.x / depthFactor,
        .height = windowSize.y / depthFactor,
        .internalFormat = GL_R16,
        .format = GL_RED,
        .type = GL_UNSIGNED_SHORT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    }, depthURL);
    PoseStreamer poseStreamer(&camera, poseURL);

    std::cout << "Video URL: " << videoURL << std::endl;
    std::cout << "Depth URL: " << depthURL << std::endl;
    std::cout << "Pose URL: " << poseURL << std::endl;

    int width = windowSize.x / surfelSize;
    int height = windowSize.y / surfelSize;

    int numVertices = width * height;

    int numTriangles = (width-1) * (height-1) * 2;
    int indexBufferSize = numTriangles * 3;

    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .diffuseTexture = &videoTextureColor }),
        .pointcloud = renderState == RenderState::POINTCLOUD,
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Mesh meshWireframe = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeWireframe = Node(&meshWireframe);
    nodeWireframe.frustumCulled = false;
    nodeWireframe.wireframe = true;
    nodeWireframe.visible = false;
    scene.addChildNode(&nodeWireframe);

    double elapsedTimeColor, elapsedTimeDepth;
    pose_id_t poseIdColor = -1, poseIdDepth = -1;
    bool mwEnabled = true;
    bool sync = true;
    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showVideoPreview = true;
        static bool showDepthPreview = true;

        glm::vec2 winSize = glm::vec2(windowSize.x, windowSize.y);

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
            ImGui::MenuItem("Video Preview", 0, &showVideoPreview);
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
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                camera.setRotationEuler(rotation);
            }

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: RGB (%.1f FPS), D (%.1f FPS)", videoTextureColor.getFrameRate(), videoTextureDepth.getFrameRate());
            ImGui::TextColored(ImVec4(1,0.5,0,1), "E2E Latency: RGB (%.1f ms), D (%.1f ms)", elapsedTimeColor, elapsedTimeDepth);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.1f ms", videoTextureColor.stats.timeToReceiveMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decode frame: %.1f ms", videoTextureColor.stats.timeToDecodeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to resize frame: %.1f ms", videoTextureColor.stats.timeToResizeMs);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Bitrate: RGB (%.1f Mbps), D (%.1f Mbps)", videoTextureColor.stats.bitrateMbps, videoTextureDepth.stats.bitrateMbps);

            ImGui::Separator();

            ImGui::Text("Remote Pose ID: RGB (%d), D (%d)", poseIdColor, poseIdDepth);

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
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - 2 * TEXTURE_PREVIEW_SIZE - 60, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Raw Color Texture", &showVideoPreview, flags);
            ImGui::Image((void*)(intptr_t)videoTextureColor.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        if (showDepthPreview) {
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - TEXTURE_PREVIEW_SIZE - 30, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Raw Depth Texture", &showDepthPreview, flags);
            ImGui::Image((void*)(intptr_t)videoTextureDepth.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
        }

        ImGui::End();

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(winSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                if (saveAsHDR) {
                    renderer.gBuffer.saveColorAsHDR(fileName + ".hdr");
                }
                else {
                    renderer.gBuffer.saveColorAsPNG(fileName + ".png");
                }
            }

            ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.resize(windowSize.x, windowSize.y);

        camera.aspect = (float)windowSize.x / (float)windowSize.y;
        camera.updateProjectionMatrix();
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

    ComputeShader genMeshFromDepthShader({
        .computeCodePath = "./shaders/genMeshFromDepth.comp"
    });
    genMeshFromDepthShader.bind();
    genMeshFromDepthShader.setVec2("screenSize", glm::vec2(windowSize.x, windowSize.y));
    genMeshFromDepthShader.setInt("surfelSize", surfelSize);

    scene.backgroundColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    // load camera view and projection matrices
    auto cameraData = FileIO::loadBinaryFile(DATA_PATH + "data/camera.bin");
    glm::mat4 proj = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    std::memcpy(&proj, cameraData.data(), sizeof(glm::mat4));
    std::memcpy(&view, cameraData.data() + sizeof(glm::mat4), sizeof(glm::mat4));

    camera.setProjectionMatrix(proj);
    camera.setViewMatrix(view);

    // load remote camera
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    auto remoteCameraData = FileIO::loadBinaryFile(DATA_PATH + "data/remoteCamera.bin");
    std::memcpy(&proj, remoteCameraData.data(), sizeof(glm::mat4));
    std::memcpy(&view, remoteCameraData.data() + sizeof(glm::mat4), sizeof(glm::mat4));
    remoteCamera.setProjectionMatrix(proj);
    remoteCamera.setViewMatrix(view);

    Pose currentColorFramePose, currentDepthFramePose;

    std::vector<Vertex> newVertices(numVertices);
    std::vector<unsigned int> newIndices(indexBufferSize);
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

        // send pose to streamer
        poseStreamer.sendPose();

        // render color video frame
        videoTextureColor.bind();
        poseIdColor = videoTextureColor.draw();
        videoTextureColor.unbind();

        // render depth video frame
        videoTextureDepth.bind();
        if (sync) {
            poseIdDepth = videoTextureDepth.draw(poseIdColor);
        }
        else {
            poseIdDepth = videoTextureDepth.draw();
        }
        videoTextureDepth.unbind();

        if (!mwEnabled) {
            if (poseIdColor != -1) poseStreamer.getPose(poseIdColor, &currentColorFramePose, &elapsedTimeColor);

            videoShader.bind();
            videoShader.setTexture("tex", videoTextureColor, 5);
            renderStats = renderer.drawToScreen(videoShader);

            return;
        }

        // set shader uniforms
        genMeshFromDepthShader.bind();
        {
            genMeshFromDepthShader.setMat4("projection", remoteCamera.getProjectionMatrix());
            genMeshFromDepthShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
            genMeshFromDepthShader.setFloat("near", remoteCamera.near);
            genMeshFromDepthShader.setFloat("far", remoteCamera.far);
        }
        {
            genMeshFromDepthShader.setTexture(videoTextureDepth, 0);
        }
        {
            if (poseStreamer.getPose(poseIdColor, &currentColorFramePose, &elapsedTimeColor)) {
                genMeshFromDepthShader.setMat4("viewColor", currentColorFramePose.mono.view);
            }
            if (poseStreamer.getPose(poseIdDepth, &currentDepthFramePose, &elapsedTimeDepth)) {
                genMeshFromDepthShader.setMat4("viewInverseDepth", glm::inverse(currentDepthFramePose.mono.view));
            }
        }
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, meshWireframe.vertexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, meshWireframe.indexBuffer);
        }
        // dispatch compute shader to generate vertices and indices for mesh
        genMeshFromDepthShader.dispatch(width / 16, height / 16, 1);
        genMeshFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        poseStreamer.removePosesLessThan(std::min(poseIdColor, poseIdDepth));

        // set render state
        mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        nodeWireframe.visible = renderState == RenderState::WIREFRAME;

        nodeWireframe.setPosition(node.getPosition() - camera.getForwardVector() * 0.001f);

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        screenShader.bind();
        screenShader.setBool("doToneMapping", false); // video is already tone mapped
        renderer.drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
