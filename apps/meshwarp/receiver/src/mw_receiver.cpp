#include <iostream>

#include <args.hxx>
#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
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

#include <VideoTexture.h>
#include <DepthReceiverTexture.h>
#include <PoseStreamer.h>

#define TEXTURE_PREVIEW_SIZE 500

#define VERTICES_IN_A_QUAD 4

const std::string DATA_PATH = "../streamer/";

struct ResultVertex {
    glm::vec3 position;
    int padding1;
    glm::vec2 texCoords;
    int padding2;
    int padding3;
};

enum class RenderState {
    MESH,
    POINTCLOUD,
    WIREFRAME
};

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Reciever";
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;

    RenderState renderState = RenderState::MESH;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'i', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 8);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "Depth URL", {'e', "depth-url"}, "0.0.0.0:65432");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "Pose URL", {'p', "pose-url"}, "127.0.0.1:54321");
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
    std::string videoURL = args::get(videoURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string poseURL = args::get(poseURLIn);

    int surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

    VideoTexture videoTextureColor({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_SRGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL);
    DepthReceiverTexture videoTextureDepth({
        .width = screenWidth,
        .height = screenHeight,
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

    int trianglesDrawn = 0;
    double elapsedTime = 0.0f;
    bool disableMeshWarp;
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

        ImGui::InputFloat3("Camera Position", (float*)&camera.position);
        ImGui::SliderFloat("Movement Speed", &camera.movementSpeed, 0.1f, 20.0f);

        ImGui::Separator();

        ImGui::Text("Video URL: %s", videoURL.c_str());
        ImGui::Text("Pose URL: %s", poseURL.c_str());

        ImGui::Separator();

        ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoTextureColor.getFrameRate(), 1000.0f / videoTextureColor.getFrameRate());
        ImGui::TextColored(ImVec4(1,0.5,0,1), "E2E Latency: %.1f ms", elapsedTime);

        ImGui::Separator();

        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms", videoTextureColor.stats.timeToReceiveFrame);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decode frame: %.3f ms", videoTextureColor.stats.timeToDecode);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to resize frame: %.3f ms", videoTextureColor.stats.timeToResize);

        ImGui::Separator();

        ImGui::Checkbox("Disable Mesh Warp", &disableMeshWarp);

        ImGui::Separator();

        ImGui::RadioButton("Render Mesh", (int*)&renderState, 0);
        ImGui::RadioButton("Render Point Cloud", (int*)&renderState, 1);
        ImGui::RadioButton("Render Wireframe", (int*)&renderState, 2);

        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(screenWidth - 2 * TEXTURE_PREVIEW_SIZE - 60, 10), ImGuiCond_FirstUseEver);
        flags = ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::Begin("Raw Color Texture", 0, flags);
        ImGui::Image((void*)(intptr_t)videoTextureColor.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Raw Depth Texture", 0, flags);
        ImGui::Image((void*)(intptr_t)videoTextureDepth.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));

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

    Shader videoShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayTexture.frag",
    });

    ComputeShader genMeshShader({
        .computeCodePath = "./shaders/genMesh.comp"
    });

    int width = screenWidth / surfelSize;
    int height = screenHeight / surfelSize;

    GLuint vertexBuffer;
    int numVertices = width * height * VERTICES_IN_A_QUAD;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(ResultVertex), nullptr, GL_STATIC_DRAW);

    GLuint indexBuffer;
    int numTriangles = width * height * 2;
    int indexBufferSize = numTriangles * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    genMeshShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    genMeshShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genMeshShader.setInt("surfelSize", surfelSize);
    genMeshShader.unbind();

    Mesh mesh = Mesh({
        .vertices = {},
        .indices = {},
        .material = new UnlitMaterial({ .diffuseTextureID = videoTextureColor.ID }),
        .wireframe = false,
        .pointcloud = renderState == RenderState::POINTCLOUD,
    });
    Node node = Node(&mesh);
    scene.addChildNode(&node);

    Mesh wireframeMesh = Mesh({
        .vertices = {},
        .indices = {},
        .material = new UnlitMaterial({ .color = glm::vec3(1.0f, 1.0f, 0.0f) }),
        .wireframe = true,
        .pointcloud = false,
    });
    Node wireframeNode = Node(&wireframeMesh);
    wireframeNode.setTranslation(glm::vec3(0.0f, 0.001f, 0.001f));
    scene.addChildNode(&wireframeNode);

    scene.backgroundColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    // load camera view and projection matrices
    std::ifstream cameraFile(DATA_PATH + "data/camera.bin", std::ios::binary);
    glm::mat4 proj = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    cameraFile.read(reinterpret_cast<char*>(&proj), sizeof(glm::mat4));
    cameraFile.read(reinterpret_cast<char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    camera.setProjectionMatrix(proj);
    camera.setViewMatrix(view);

    // load remote camera
    Camera remoteCamera = Camera(screenWidth, screenHeight);
    std::ifstream remoteCameraFile(DATA_PATH + "data/remoteCamera.bin", std::ios::binary);
    remoteCameraFile.read(reinterpret_cast<char*>(&proj), sizeof(glm::mat4));
    remoteCameraFile.read(reinterpret_cast<char*>(&view), sizeof(glm::mat4));
    remoteCamera.setProjectionMatrix(proj);
    remoteCamera.setViewMatrix(view);

    pose_id_t poseIdColor, poseIdDepth;
    Pose currentColorFramePose, currentDepthFramePose;
    double elapsedTimeColor, elapsedTimeDepth;
    std::vector<Vertex> newVertices(numVertices);
    std::vector<unsigned int> newIndices(indexBufferSize);
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

        // send pose to streamer
        poseStreamer.sendPose();

        // render color video frame
        videoTextureColor.bind();
        poseIdColor = videoTextureColor.draw();
        videoTextureColor.unbind();

        // render depth video frame
        videoTextureDepth.bind();
        poseIdDepth = videoTextureDepth.draw();
        videoTextureDepth.unbind();

        if (disableMeshWarp) {
            if (poseIdColor != -1) poseStreamer.getPose(poseIdColor, &currentColorFramePose, &elapsedTime);

            videoShader.bind();
            videoShader.setInt("tex", 5);
            videoTextureColor.bind(5);
            app.renderer->drawToScreen(videoShader);

            return;
        }

        // set shader uniforms
        genMeshShader.bind();
        genMeshShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        genMeshShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
        genMeshShader.setFloat("near", remoteCamera.near);
        genMeshShader.setFloat("far", remoteCamera.far);
        videoTextureDepth.bind(0);
        if (poseIdColor != -1 && poseStreamer.getPose(poseIdColor, &currentColorFramePose, &elapsedTimeColor)) {
            genMeshShader.setMat4("viewColor", currentColorFramePose.view);
        }
        if (poseIdDepth != -1 && poseStreamer.getPose(poseIdDepth, &currentDepthFramePose, &elapsedTimeDepth)) {
            genMeshShader.setMat4("viewInverseDepth", glm::inverse(currentDepthFramePose.view));
        }
        elapsedTime = std::fmax(elapsedTimeColor, elapsedTimeDepth);

        // dispatch compute shader
        genMeshShader.dispatch(width, height, 1);
        genMeshShader.unbind();

        if (poseIdDepth != -1 && poseIdColor != -1) {
            poseStreamer.removePosesLessThan(std::min(poseIdColor, poseIdDepth));
        }

        // create mesh from compute shader output
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
        GLvoid* pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (pBuffer) {
            ResultVertex* pVertices = static_cast<ResultVertex*>(pBuffer);

            int x, y;
            for (int i = 0; i < numVertices; i+=VERTICES_IN_A_QUAD) {
                x = (i / VERTICES_IN_A_QUAD) % width;
                y = (i / VERTICES_IN_A_QUAD) / width;

                Vertex vertexUpperLeft;
                vertexUpperLeft.position = glm::vec3(pVertices[i+0].position);
                vertexUpperLeft.texCoords = glm::vec2(pVertices[i+0].texCoords);

                Vertex vertexUpperRight;
                vertexUpperRight.position = glm::vec3(pVertices[i+1].position);
                vertexUpperRight.texCoords = glm::vec2(pVertices[i+1].texCoords);

                Vertex vertexLowerLeft;
                vertexLowerLeft.position = glm::vec3(pVertices[i+2].position);
                vertexLowerLeft.texCoords = glm::vec2(pVertices[i+2].texCoords);

                Vertex vertexLowerRight;
                vertexLowerRight.position = glm::vec3(pVertices[i+3].position);
                vertexLowerRight.texCoords = glm::vec2(pVertices[i+3].texCoords);

                newVertices[i+0] = vertexUpperLeft;
                newVertices[i+1] = vertexUpperRight;
                newVertices[i+2] = vertexLowerLeft;
                newVertices[i+3] = vertexLowerRight;
            }

            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        } else {
            throw std::runtime_error("Failed to map vertex buffer");
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
        GLvoid* pIndexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (pIndexBuffer) {
            memcpy(newIndices.data(), pIndexBuffer, indexBufferSize * sizeof(unsigned int));
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        } else {
            std::cerr << "Failed to save index buffer" << std::endl;
        }

        mesh.setBuffers(newVertices, newIndices);
        wireframeMesh.setBuffers(newVertices, newIndices);

        mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        wireframeMesh.visible = renderState == RenderState::WIREFRAME;

        // render all objects in scene
        trianglesDrawn = app.renderer->drawObjects(scene, camera);

        // render to screen
        screenShader.bind();
        screenShader.setBool("doToneMapping", false); // video is already tone mapped
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
