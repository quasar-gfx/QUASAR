#include <iostream>

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
#include <PoseStreamer.h>

#define VIDEO_PREVIEW_SIZE 500

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
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;

    std::string videoURL = "0.0.0.0:12345";
    std::string depthURL = "0.0.0.0:65432";
    std::string poseURL = "127.0.0.1:54321";
    int surfelSize = 4;
    RenderState renderState = RenderState::MESH;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-ss") && i + 1 < argc) {
            surfelSize = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-pc") && i + 1 < argc) {
            renderState = RenderState::POINTCLOUD;
            i++;
        }
        else if (!strcmp(argv[i], "-wf") && i + 1 < argc) {
            renderState = RenderState::WIREFRAME;
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

    VideoTexture videoTextureColor({
        .width = config.width,
        .height = config.height,
        .internalFormat = GL_SRGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, videoURL);
    VideoTexture videoTextureDepth({
        .width = config.width,
        .height = config.height,
        .internalFormat = GL_RGBA16,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    }, depthURL);
    PoseStreamer poseStreamer(&camera, poseURL);

    std::cout << "Video URL: " << videoURL << std::endl;
    std::cout << "Depth URL: " << depthURL << std::endl;
    std::cout << "Pose URL: " << poseURL << std::endl;

    int trianglesDrawn = 0;
    double elapedTime = 0.0f;
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
        ImGui::TextColored(ImVec4(1,0.5,0,1), "E2E Latency: %.1f ms", elapedTime);

        ImGui::Separator();

        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms", videoTextureColor.stats.timeToReceiveFrame);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decode frame: %.3f ms", videoTextureColor.stats.timeToDecode);
        ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to resize frame: %.3f ms", videoTextureColor.stats.timeToResize);

        ImGui::Separator();

        ImGui::RadioButton("Render Mesh", (int*)&renderState, 0);
        ImGui::RadioButton("Render Point Cloud", (int*)&renderState, 1);
        ImGui::RadioButton("Render Wireframe", (int*)&renderState, 2);

        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(screenWidth - 2 * VIDEO_PREVIEW_SIZE - 60, 10), ImGuiCond_FirstUseEver);
        flags = ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::Begin("Raw Color Texture", 0, flags);
        ImGui::Image((void*)(intptr_t)videoTextureColor.ID, ImVec2(VIDEO_PREVIEW_SIZE, VIDEO_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(screenWidth - VIDEO_PREVIEW_SIZE - 30, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Raw Depth Texture", 0, flags);
        ImGui::Image((void*)(intptr_t)videoTextureDepth.ID, ImVec2(VIDEO_PREVIEW_SIZE, VIDEO_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));

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

    ComputeShader genMeshShader({
        .computeCodePath = "./shaders/genMesh.comp"
    });

    int width = screenWidth / surfelSize;
    int height = screenHeight / surfelSize;

    GLuint vertexBuffer;
    int numVertices = width * height * VERTICES_IN_A_QUAD;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);

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

    pose_id_t colorPoseID, depthPoseID;
    Pose currentFramePose;
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

        colorPoseID = videoTextureColor.getLatestPoseID();
        depthPoseID = videoTextureDepth.getLatestPoseID();

        // color is behind depth
        if (colorPoseID < depthPoseID) {
            // render color video frame
            videoTextureColor.bind();
            colorPoseID = videoTextureColor.draw();
            videoTextureColor.unbind();

            // render depth video frame
            videoTextureDepth.bind();
            depthPoseID = videoTextureDepth.draw(colorPoseID);
            videoTextureDepth.unbind();
        }
        // depth is behind color
        else {
            // render depth video frame
            videoTextureDepth.bind();
            depthPoseID = videoTextureDepth.draw();
            videoTextureDepth.unbind();

            // render color video frame
            videoTextureColor.bind();
            colorPoseID = videoTextureColor.draw(depthPoseID);
            videoTextureColor.unbind();
        }

        if (depthPoseID != -1 && colorPoseID == depthPoseID) {
            genMeshShader.bind();
            if (poseStreamer.getPose(depthPoseID, &currentFramePose, &elapedTime)) {
                genMeshShader.setMat4("viewInverse", glm::inverse(currentFramePose.view));
                genMeshShader.setMat4("projectionInverse", glm::inverse(remoteCamera.getProjectionMatrix()));
                genMeshShader.setFloat("near", remoteCamera.near);
                genMeshShader.setFloat("far", remoteCamera.far);
            }
            videoTextureDepth.bind(0);
            genMeshShader.dispatch(width, height, 1);
            genMeshShader.unbind();

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
            GLvoid* pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pBuffer) {
                glm::vec4* pVertices = static_cast<glm::vec4*>(pBuffer);

                int x, y;
                for (int i = 0; i < numVertices; i+=VERTICES_IN_A_QUAD) {
                    x = (i / VERTICES_IN_A_QUAD) % width;
                    y = (i / VERTICES_IN_A_QUAD) / width;

                    Vertex vertexUpperLeft;
                    vertexUpperLeft.position = glm::vec3(pVertices[i+0]);
                    vertexUpperLeft.texCoords = glm::vec2((float)x / (float)(width), (float)(y + 1) / (float)(height));

                    Vertex vertexUpperRight;
                    vertexUpperRight.position = glm::vec3(pVertices[i+1]);
                    vertexUpperRight.texCoords = glm::vec2((float)(x + 1) / (float)(width), (float)(y + 1) / (float)(height));

                    Vertex vertexLowerLeft;
                    vertexLowerLeft.position = glm::vec3(pVertices[i+2]);
                    vertexLowerLeft.texCoords = glm::vec2((float)x / (float)(width), (float)y / (float)(height));

                    Vertex vertexLowerRight;
                    vertexLowerRight.position = glm::vec3(pVertices[i+3]);
                    vertexLowerRight.texCoords = glm::vec2((float)(x + 1) / (float)(width), (float)y / (float)(height));

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
        }

        mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        wireframeMesh.visible = renderState == RenderState::WIREFRAME;

        // render all objects in scene
        trianglesDrawn = app.renderer->drawObjects(scene, camera);

        // render to screen
        app.renderer->drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    videoTextureColor.cleanup();
    videoTextureDepth.cleanup();

    return 0;
}
