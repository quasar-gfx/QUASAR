#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
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

#include <VideoStreamer.h>
#include <PoseReceiver.h>

#define VERTICES_IN_A_QUAD 4

int main(int argc, char** argv) {
    Config config{};
    config.title = "MeshWarp Streamer";
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;
    config.showWindow = false;

    std::string scenePath = "../assets/scenes/sponza.json";
    int surfelSize = 4;
    std::string videoURL = "127.0.0.1:12345";
    std::string depthURL = "127.0.0.1:65432";
    std::string poseURL = "0.0.0.0:54321";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            scenePath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            config.showWindow = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-ss") && i + 1 < argc) {
            surfelSize = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            videoURL = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            poseURL = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) {
            depthURL = argv[i + 1];
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
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, scene, camera);

    RenderTarget renderTargetColor({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_RGBA16,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    });

    RenderTarget renderTargetDepth({
        .width = 2 * screenWidth,
        .height = screenHeight,
        .internalFormat = GL_RGBA16,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    });

    VideoStreamer videoStreamerColor = VideoStreamer(&renderTargetColor, videoURL);
    VideoStreamer videoStreamerDepth = VideoStreamer(&renderTargetDepth, depthURL);
    PoseReceiver poseReceiver = PoseReceiver(&camera, poseURL);

    std::cout << "Video URL: " << videoURL << std::endl;
    std::cout << "Depth URL: " << depthURL << std::endl;
    std::cout << "Pose URL: " << poseURL << std::endl;

    bool paused = false;
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

        ImGui::Text("Video URL: %s", videoURL.c_str());
        ImGui::Text("Pose URL: %s", poseURL.c_str());

        ImGui::Separator();

        ImGui::TextColored(ImVec4(1,0.5,0,1), "Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamerColor.getFrameRate(), 1000.0f / videoStreamerColor.getFrameRate());

        ImGui::Separator();

        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: %.3f ms", videoStreamerColor.stats.timeToCopyFrame);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: %.3f ms", videoStreamerColor.stats.timeToEncode);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: %.3f ms", videoStreamerColor.stats.timeToSendFrame);

        ImGui::Separator();

        ImGui::Checkbox("Pause", &paused);

        ImGui::End();
    });

    // shaders
    Shader colorShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader depthShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayDepth.frag"
    });

    ComputeShader genMeshShader({
        .computeCodePath = "shaders/genMesh.comp"
    });

    int width = screenWidth / surfelSize;
    int height = screenHeight / surfelSize;

    GLuint vertexBuffer;
    int numVertices = width * height * VERTICES_IN_A_QUAD;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);

    GLuint indexBuffer;
    int trianglesDrawn = width * height * 2;
    int indexBufferSize = trianglesDrawn * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    genMeshShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    genMeshShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genMeshShader.unbind();

    // save camera view and projection matrices
    std::ofstream cameraFile;
    cameraFile.open("data/camera.bin", std::ios::out | std::ios::binary);
    glm::mat4 proj = camera.getProjectionMatrix();
    glm::mat4 view = camera.getViewMatrix();
    cameraFile.write(reinterpret_cast<const char*>(&proj), sizeof(glm::mat4));
    cameraFile.write(reinterpret_cast<const char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    // set high fov
    camera.setFovy(glm::radians(100.0f));

    pose_id_t poseId = 0;
    bool firstFrame = true;
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

        if (paused) {
            return;
        }

        // receive pose
        poseId = poseReceiver.receivePose(false);

        // render all objects in scene
        app.renderer->drawObjects(scene, camera);

        // render to screen
        // app.renderer->drawToScreen(colorShader);
        app.renderer->drawToRenderTarget(colorShader, renderTargetColor);
        depthShader.bind();
        depthShader.setFloat("near", camera.near);
        depthShader.setFloat("far", camera.far);
        app.renderer->drawToRenderTarget(depthShader, renderTargetDepth);

        // send video frame
        if (poseId != -1) {
            videoStreamerColor.sendFrame(poseId);
            videoStreamerDepth.sendFrame(poseId);
        }

        // if (firstFrame) {
        //     double startTime = glfwGetTime();

        //     genMeshShader.bind();
        //     genMeshShader.setMat4("viewInverse", glm::inverse(camera.getViewMatrix()));
        //     genMeshShader.setMat4("projectionInverse", glm::inverse(camera.getProjectionMatrix()));
        //     genMeshShader.setFloat("near", camera.near);
        //     genMeshShader.setFloat("far", camera.far);
        //     genMeshShader.setInt("surfelSize", surfelSize);
        //     app.renderer->gBuffer.positionBuffer.bind(0);
        //     app.renderer->gBuffer.normalsBuffer.bind(1);
        //     app.renderer->gBuffer.depthBuffer.bind(2);
        //     genMeshShader.dispatch(width, height, 1);
        //     genMeshShader.unbind();

        //     std::cout << "\tRendering Time: " << glfwGetTime() - startTime << "s" << std::endl;
        //     // startTime = glfwGetTime();

        //     // renderTarget.bind();
        //     // renderTarget.colorBuffer.saveTextureToPNG("imgs/color_center_0.png");
        //     // renderTarget.unbind();

        //     // std::cout << "\tSaving Texture Time: " << glfwGetTime() - startTime << "s" << std::endl;
        //     // startTime = glfwGetTime();

        //     // std::ofstream depthFile;
        //     // depthFile.open("data/depth_center_0.bin", std::ios::out | std::ios::binary);

        //     std::ofstream positionsFile;
        //     positionsFile.open("data/positions_center_0.bin", std::ios::out | std::ios::binary);

        //     std::ofstream indicesFile;
        //     indicesFile.open("data/indices_center_0.bin", std::ios::out | std::ios::binary);

        //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
        //     GLvoid* pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        //     if (pBuffer) {
        //         glm::vec4* pVertices = static_cast<glm::vec4*>(pBuffer);

        //         for (int i = 0; i < numVertices; i++) {
        //             Vertex vertex;
        //             vertex.position.x = pVertices[i].x;
        //             vertex.position.y = pVertices[i].y;
        //             vertex.position.z = pVertices[i].z;

        //             // std::cout << "Vertex: " << vertex.position.x << ", " << vertex.position.y << ", " << vertex.position.z << std::endl;

        //             positionsFile.write(reinterpret_cast<const char*>(&vertex.position), sizeof(glm::vec3));
        //             // depthFile.write(reinterpret_cast<const char*>(&pVertices[i].w), sizeof(pVertices[i].w));
        //         }

        //         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        //     } else {
        //         std::cerr << "Failed to save vertex buffer" << std::endl;
        //     }

        //     std::cout << "\tSaving Vertices Time: " << glfwGetTime() - startTime << "s" << std::endl;
        //     std::cout << "\t" << numVertices << " vertices" << std::endl;
        //     startTime = glfwGetTime();

        //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
        //     GLvoid* pIndexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        //     if (pIndexBuffer) {
        //         indicesFile.write(reinterpret_cast<const char*>(pIndexBuffer), indexBufferSize * sizeof(unsigned int));
        //         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        //     } else {
        //         std::cerr << "Failed to save index buffer" << std::endl;
        //     }

        //     std::cout << "\tSaving Indices Time: " << glfwGetTime() - startTime << "s" << std::endl;

        //     firstFrame = false;
        // }
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    return 0;
}
