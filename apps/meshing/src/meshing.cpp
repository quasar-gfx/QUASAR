#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Primatives/Model.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <SceneLoader.h>

const std::string BACKPACK_MODEL_PATH = "../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Meshing Test";
    app.config.openglMajorVersion = 4;
    app.config.openglMinorVersion = 3;
    app.config.enableVSync = false;
    app.config.showWindow = false;

    std::string scenePath = "../assets/scenes/sponza.json";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            app.config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            app.config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            scenePath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            app.config.showWindow = atoi(argv[i + 1]);
            i++;
        }
    }

    GLFWWindow window(app.config);
    app.init(&window);

    unsigned int screenWidth, screenHeight;
    window.getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, scene, camera);

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();
    });

    // shaders
    Shader screenShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    ComputeShader genMeshShader({
        .computeCodePath = "shaders/genMesh.comp"
    });

    GLuint vertexBuffer;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, screenWidth * screenHeight * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);

    GLuint indexBuffer;
    int numTriangles = screenWidth * screenHeight * 2;
    int indexBufferSize = numTriangles * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    GLuint texCoordBuffer;
    glGenBuffers(1, &texCoordBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, texCoordBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, screenWidth * screenHeight * sizeof(glm::vec2), nullptr, GL_STATIC_DRAW);

    genMeshShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, texCoordBuffer);
    genMeshShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genMeshShader.unbind();

    // camera.setProjectionMatrix(glm::radians(120.0f), (float)screenWidth / (float)screenHeight, 0.1f, 1000.0f);

    Framebuffer outputFramebuffer;
    outputFramebuffer.createColorAndDepthBuffers(screenWidth, screenHeight);

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
            std::cout << "saving " << label << " t=" << std::to_string(timestamp) << std::endl;

            double startTime = glfwGetTime();

            camera.position = position;
            camera.updateViewMatrix();

            // render all objects in scene
            app.renderer.drawObjects(scene, camera);

            genMeshShader.bind();
            genMeshShader.setMat4("viewInverse", glm::inverse(camera.getViewMatrix()));
            genMeshShader.setMat4("projectionInverse", glm::inverse(camera.getProjectionMatrix()));
            genMeshShader.setFloat("near", camera.near);
            genMeshShader.setFloat("far", camera.far);
            glBindImageTexture(0, app.renderer.gBuffer.positionBuffer.ID, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
            glBindImageTexture(1, app.renderer.gBuffer.normalsBuffer.ID, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
            glBindImageTexture(2, app.renderer.gBuffer.colorBuffer.ID, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA);
            app.renderer.gBuffer.depthBuffer.bind(3);
            genMeshShader.dispatch(screenWidth / 10, screenHeight / 10, 1);
            genMeshShader.unbind();

            // render to screen
            // app.renderer.drawToScreen(screenShader);
            app.renderer.drawToFramebuffer(screenShader, outputFramebuffer);

            std::cout << "\tRendering Time: " << glfwGetTime() - startTime << "s" << std::endl;

            // app.renderer.gBuffer.colorBuffer.saveTextureToPNG("imgs/color_" + label + "_" + std::to_string(timestamp) + ".png");
            // app.renderer.gBuffer.depthBuffer.saveDepthToFile("imgs/depth1.bin");
            outputFramebuffer.bind();
            outputFramebuffer.colorBuffer.saveTextureToPNG("imgs/color_" + label + "_" + std::to_string(timestamp) + ".png");
            outputFramebuffer.unbind();

            std::ofstream depthFile;
            depthFile.open("data/depth_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            std::ofstream positionsFile;
            positionsFile.open("data/positions_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            // std::ofstream objFile;
            // objFile.open("mesh.obj");

            // std::ofstream mtlFile;
            // mtlFile.open("mesh.mtl");

            std::vector<Vertex> vertices;
            vertices.reserve(screenWidth * screenHeight);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
            GLvoid* pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pBuffer) {
                glm::vec4* pVertices = static_cast<glm::vec4*>(pBuffer);

                for (int i = 0; i < screenWidth * screenHeight; i++) {
                    Vertex vertex;
                    vertex.position.x = pVertices[i].x;
                    vertex.position.y = pVertices[i].y;
                    vertex.position.z = pVertices[i].z;
                    vertices.push_back(vertex);

                    positionsFile.write(reinterpret_cast<const char*>(&pVertices[i].x), 3 * sizeof(float));
                    depthFile.write(reinterpret_cast<const char*>(&pVertices[i].w), sizeof(pVertices[i].w));
                }

                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to map vertex buffer into memory" << std::endl;
            }

/*             std::vector<unsigned int> indices;
 *             indices.reserve(indexBufferSize);
 *             glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
 *             GLvoid* pIndexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
 *             if (pIndexBuffer) {
 *                 unsigned int* pIndices = static_cast<unsigned int*>(pIndexBuffer);
 *
 *                 for (int i = 0; i < indexBufferSize; i++) {
 *                     indices.push_back(pIndices[i]);
 *                 }
 *
 *                 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 *             } else {
 *                 std::cerr << "Failed to map index buffer into memory" << std::endl;
 *             }
 *
 *             std::vector<glm::vec2> texCoords;
 *             texCoords.reserve(screenWidth * screenHeight);
 *             glBindBuffer(GL_SHADER_STORAGE_BUFFER, texCoordBuffer);
 *             GLvoid* pTexCoordBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
 *             if (pTexCoordBuffer) {
 *                 glm::vec2* pTexCoords = static_cast<glm::vec2*>(pTexCoordBuffer);
 *
 *                 for (int i = 0; i < screenWidth * screenHeight; i++) {
 *                     texCoords.push_back(pTexCoords[i]);
 *                 }
 *
 *                 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 *             } else {
 *                 std::cerr << "Failed to map tex coord buffer into memory" << std::endl;
 *             }
 *
 *             if (objFile.is_open()) {
 *                 objFile << "mtllib " << "mesh.mtl" << std::endl;
 *                 objFile << "usemtl " << "Material1" << std::endl;
 *                 for (int i = 0; i < vertices.size(); i++) {
 *                     objFile << "v " << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
 *                     positionsFile.write(reinterpret_cast<const char*>(&vertices[i].position.x), sizeof(vertices[i].position.x));
 *                     positionsFile.write(reinterpret_cast<const char*>(&vertices[i].position.y), sizeof(vertices[i].position.y));
 *                     positionsFile.write(reinterpret_cast<const char*>(&vertices[i].position.z), sizeof(vertices[i].position.z));
 *                     // positionsFile << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
 *                 }
 *                 for (int i = 0; i < texCoords.size(); i++) {
 *                     objFile << "vt " << texCoords[i].x << " " << texCoords[i].y << std::endl;
 *                 }
 *                 // for (int i = 0; i < indices.size(); i += 3) {
 *                 //     objFile << "f " << indices[i] + 1 << " " << indices[i + 1] + 1 << " " << indices[i + 2] + 1<< std::endl;
 *                 // }
 *                 for (int i = 0; i < indices.size(); i += 3) {
 *                     objFile << "f " << indices[i + 0] + 1 << "/" << indices[i + 0] + 1 << " " << indices[i + 1] + 1 << "/" << indices[i + 1] + 1 << " " << indices[i + 2] + 1 << "/" << indices[i + 2] + 1 << std::endl;
 *                 }
 *                 objFile.close();
 *             } else {
 *                 std::cerr << "Failed to open objFile" << std::endl;
 *             }
 *
 *             if (mtlFile.is_open()) {
 *                 mtlFile << "newmtl " << "Material1" << std::endl;
 *                 mtlFile << "map_Kd " << "meshColor.png" << std::endl;
 *             } */

            std::cout << "\tSaving Time: " << glfwGetTime() - startTime << "s" << std::endl;
        };

        saveFrame(initialPosition + glm::vec3(0.0f, 0.0f, 0.0f + z), "center", t);
        saveFrame(initialPosition + glm::vec3(0.5f, 0.5f, 0.5f + z), "top_right_front", t);
        saveFrame(initialPosition + glm::vec3(0.5f, 0.5f, -0.5f + z), "top_right_back", t);
        saveFrame(initialPosition + glm::vec3(-0.5f, 0.5f, 0.5f + z), "top_left_front", t);
        saveFrame(initialPosition + glm::vec3(-0.5f, 0.5f, -0.5f + z), "top_left_back", t);
        saveFrame(initialPosition + glm::vec3(0.5f, -0.5f, 0.5f + z), "bottom_right_front", t);
        saveFrame(initialPosition + glm::vec3(0.5f, -0.5f, -0.5f + z), "bottom_right_back", t);
        saveFrame(initialPosition + glm::vec3(-0.5f, -0.5f, 0.5f + z), "bottom_left_front", t);
        saveFrame(initialPosition + glm::vec3(-0.5f, -0.5f, -0.5f + z), "bottom_left_back", t);

        z -= 0.5f;

        t++;
        if (t >= 10) {
            window.close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
