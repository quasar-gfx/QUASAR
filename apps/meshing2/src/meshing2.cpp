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

#define VERTICES_IN_A_QUAD 4

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Test";
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;
    config.enableVSync = false;
    config.showWindow = false;

    int maxSteps = 10;
    int surfelSize = 4;
    std::string scenePath = "../assets/scenes/sponza.json";
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
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            config.showWindow = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            maxSteps = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-ss") && i + 1 < argc) {
            surfelSize = atoi(argv[i + 1]);
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
    bool res = loader.loadScene(scenePath, scene, camera);
    if (!res) {
        std::cerr << "Failed to load scene: " << scenePath << std::endl;
        return 1;
    }

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
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::End();
    });

    // shaders
    Shader screenShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        // .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
        .fragmentCodePath = "../shaders/postprocessing/displayIDs.frag"
    });

    ComputeShader genMesh2Shader({
        .computeCodePath = "shaders/genMesh2.comp"
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

    // GLuint texCoordBuffer;
    // glGenBuffers(1, &texCoordBuffer);
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, texCoordBuffer);
    // glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(glm::vec2), nullptr, GL_STATIC_DRAW);

    genMesh2Shader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, texCoordBuffer);
    genMesh2Shader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genMesh2Shader.unbind();

    RenderTarget renderTarget({
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

    // camera.setProjectionMatrix(glm::radians(120.0f), (float)screenWidth / (float)screenHeight, 0.1f, 1000.0f);

    // save camera view and projection matrices
    std::ofstream cameraFile;
    cameraFile.open("data/camera.bin", std::ios::out | std::ios::binary);
    glm::mat4 proj = camera.getProjectionMatrix();
    glm::mat4 view = camera.getViewMatrix();
    cameraFile.write(reinterpret_cast<const char*>(&proj), sizeof(glm::mat4));
    cameraFile.write(reinterpret_cast<const char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    glm::vec3 initialPosition = camera.position;

    unsigned int t = 0;
    float z = 0.0f;
    app.onRender([&](double now, double dt) {
        auto saveFrame = [&](glm::vec3 position, std::string label, unsigned int timestamp) {
            std::cout << "saving [" << label << "] t=" << std::to_string(timestamp) << std::endl;

            double startTime = glfwGetTime();

            camera.position = position;
            camera.updateViewMatrix();

            // render all objects in scene
            app.renderer->drawObjects(scene, camera);

            genMesh2Shader.bind();
            genMesh2Shader.setMat4("viewInverse", glm::inverse(camera.getViewMatrix()));
            genMesh2Shader.setMat4("projectionInverse", glm::inverse(camera.getProjectionMatrix()));
            genMesh2Shader.setFloat("near", camera.near);
            genMesh2Shader.setFloat("far", camera.far);
            genMesh2Shader.setInt("surfelSize", surfelSize);
            glBindImageTexture(0, app.renderer->gBuffer.positionBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
            glBindImageTexture(1, app.renderer->gBuffer.normalsBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
            glBindImageTexture(2, app.renderer->gBuffer.idBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
            app.renderer->gBuffer.depthBuffer.bind(3);
            genMesh2Shader.dispatch(width, height, 1);
            genMesh2Shader.unbind();

            // render to screen
            // app.renderer->drawToScreen(screenShader);
            app.renderer->drawToRenderTarget(screenShader, renderTarget);

            std::cout << "\tRendering Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            // app.renderer->gBuffer.colorBuffer.saveTextureToPNG("imgs/color_" + label + "_" + std::to_string(timestamp) + ".png");
            // app.renderer->gBuffer.depthBuffer.saveDepthToFile("imgs/depth1.bin");
            renderTarget.bind();
            renderTarget.colorBuffer.saveTextureToPNG("imgs/color_" + label + "_" + std::to_string(timestamp) + ".png");
            renderTarget.unbind();

            std::cout << "\tSaving Texture Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            // std::ofstream depthFile;
            // depthFile.open("data/depth_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            std::ofstream positionsFile;
            positionsFile.open("data/positions_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            // std::ofstream texCoordsFile;
            // texCoordsFile.open("data/tex_coords_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            std::ofstream indicesFile;
            indicesFile.open("data/indices_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
            GLvoid* pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pBuffer) {
                glm::vec4* pVertices = static_cast<glm::vec4*>(pBuffer);

                for (int i = 0; i < numVertices; i++) {
                    Vertex vertex;
                    vertex.position.x = pVertices[i].x;
                    vertex.position.y = pVertices[i].y;
                    vertex.position.z = pVertices[i].z;

                    // std::cout << "Vertex: " << vertex.position.x << ", " << vertex.position.y << ", " << vertex.position.z << std::endl;

                    positionsFile.write(reinterpret_cast<const char*>(&vertex.position), sizeof(glm::vec3));
                    // depthFile.write(reinterpret_cast<const char*>(&pVertices[i].w), sizeof(pVertices[i].w));
                }

                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to save vertex buffer" << std::endl;
            }

            std::cout << "\tSaving Vertices Time: " << glfwGetTime() - startTime << "s" << std::endl;
            std::cout << "\t" << numVertices << " vertices" << std::endl;
            startTime = glfwGetTime();

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
            GLvoid* pIndexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pIndexBuffer) {
                indicesFile.write(reinterpret_cast<const char*>(pIndexBuffer), indexBufferSize * sizeof(unsigned int));
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to save index buffer" << std::endl;
            }

            std::cout << "\tSaving Indices Time: " << glfwGetTime() - startTime << "s" << std::endl;
        };

        saveFrame(initialPosition + glm::vec3(0.0f, 0.0f, 0.0f + z), "center", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, 0.5f, 0.5f + z), "top_right_front", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, 0.5f, -0.5f + z), "top_right_back", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, 0.5f, 0.5f + z), "top_left_front", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, 0.5f, -0.5f + z), "top_left_back", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, -0.5f, 0.5f + z), "bottom_right_front", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, -0.5f, -0.5f + z), "bottom_right_back", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, -0.5f, 0.5f + z), "bottom_left_front", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, -0.5f, -0.5f + z), "bottom_left_back", t);

        z -= 0.5f;

        t++;
        if (t >= maxSteps) {
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
