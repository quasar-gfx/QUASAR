#include <iostream>

#include <imgui/imgui.h>

#include <Shader.h>
#include <ComputeShader.h>
#include <Texture.h>
#include <Primatives.h>
#include <Model.h>
#include <CubeMap.h>
#include <Entity.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights.h>
#include <Framebuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <GLFWWindow.h>

#include <glm/gtx/string_cast.hpp>

const std::string BACKPACK_MODEL_PATH = "../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Meshing Test";
    app.config.openglMajorVersion = 4;
    app.config.openglMinorVersion = 3;

    std::string modelPath = "../assets/models/Sponza/Sponza.gltf";
    std::string hdrImagePath = "../assets/textures/hdr/barcelona.hdr";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            app.config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            app.config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            modelPath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            hdrImagePath = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            app.config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
    }

    GLFWWindow window(app.config);
    app.init(&window);

    unsigned int screenWidth, screenHeight;
    window.getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);

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
    Shader pbrShader({
        .vertexCodePath = "../assets/shaders/pbr/pbr.vert",
        .fragmentCodePath = "../assets/shaders/pbr/pbr.frag"
    });

    // converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader({
        .vertexCodePath = "../assets/shaders/cubemap/cubemap.vert",
        .fragmentCodePath = "../assets/shaders/cubemap/equirectangular2cubemap.frag"
    });

    // solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader({
        .vertexCodePath = "../assets/shaders/cubemap/cubemap.vert",
        .fragmentCodePath = "../assets/shaders/pbr/irradianceConvolution.frag"
    });

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader({
        .vertexCodePath = "../assets/shaders/cubemap/cubemap.vert",
        .fragmentCodePath = "../assets/shaders/pbr/prefilter.frag"
    });

    // BRDF shader
    Shader brdfShader({
        .vertexCodePath = "../assets/shaders/pbr/brdf.vert",
        .fragmentCodePath = "../assets/shaders/pbr/brdf.frag"
    });

    // background skybox shader
    Shader backgroundShader({
        .vertexCodePath = "../assets/shaders/cubemap/background.vert",
        .fragmentCodePath = "../assets/shaders/cubemap/backgroundHDR.frag"
    });

    Shader dirLightShadowsShader({
        .vertexCodePath = "../assets/shaders/shadows/dirShadow.vert",
        .fragmentCodePath = "../assets/shaders/shadows/dirShadow.frag"
    });

    Shader pointLightShadowsShader({
        .vertexCodePath = "../assets/shaders/shadows/pointShadow.vert",
        .fragmentCodePath = "../assets/shaders/shadows/pointShadow.frag",
        .geometryCodePath = "../assets/shaders/shadows/pointShadow.geom"
    });

    Shader screenShader({
        .vertexCodePath = "../assets/shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../assets/shaders/postprocessing/displayColor.frag"
    });

    ComputeShader genMeshShader({
        .computeCodePath = "../assets/shaders/compute/genMesh.comp"
    });

    // materials
    Material goldMaterial = Material({
        .albedoTexturePath = "../assets/textures/pbr/gold/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/gold/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/gold/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/gold/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/gold/ao.png"
    });

    Material ironMaterial = Material({
        .albedoTexturePath = "../assets/textures/pbr/rusted_iron/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/rusted_iron/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/rusted_iron/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/rusted_iron/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/rusted_iron/ao.png"
    });

    Material plasticMaterial = Material({
        .albedoTexturePath = "../assets/textures/pbr/plastic/albedo.png",
        .normalTexturePath = "../assets/textures/pbr/plastic/normal.png",
        .metallicTexturePath = "../assets/textures/pbr/plastic/metallic.png",
        .roughnessTexturePath = "../assets/textures/pbr/plastic/roughness.png",
        .aoTexturePath = "../assets/textures/pbr/plastic/ao.png"
    });

    Material windowMaterial = Material({
        .albedoTexturePath = "../assets/textures/window.png"
    });

    // objects
    Cube cubeGold = Cube(goldMaterial);
    Node cubeNodeGold = Node(&cubeGold);
    cubeNodeGold.setTranslation(glm::vec3(-0.2f, 0.25f, -7.0f));
    cubeNodeGold.setScale(glm::vec3(0.5f));

    Cube cubeIron = Cube(ironMaterial);
    Node cubeNodeIron = Node(&cubeIron);
    cubeNodeIron.setTranslation(glm::vec3(1.5f, 0.25f, -3.0f));
    cubeNodeIron.setScale(glm::vec3(0.5f));

    Sphere sphere = Sphere(plasticMaterial);
    Node sphereNodePlastic = Node(&sphere);
    sphereNodePlastic.setTranslation(glm::vec3(1.0f, 1.5f, -8.0f));
    sphereNodePlastic.setScale(glm::vec3(0.5f));

    Plane plane = Plane(windowMaterial);
    Node planeNode = Node(&plane);
    planeNode.setTranslation(glm::vec3(0.0f, 1.5f, -7.0f));
    planeNode.setRotationEuler(glm::vec3(-90.0f, 0.0f, 0.0f));
    planeNode.setScale(glm::vec3(0.5f));

    // lights
    DirectionalLight directionalLight = DirectionalLight({
        .color = glm::vec3(0.8f, 0.8f, 0.8f),
        .direction = glm::vec3(0.0f, -1.0f, -0.3f),
        .intensity = 0.1f
    });

    PointLight pointLight1 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(-1.45f, 3.5f, -6.2f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight2 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(2.2f, 3.5f, -6.2f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight3 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(-1.45f, 3.5f, 4.89f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    PointLight pointLight4 = PointLight({
        .color = glm::vec3(0.9f, 0.9f, 1.0f),
        .initialPosition = glm::vec3(2.2f, 3.5f, 4.89f),
        .intensity = 100.0f,
        .constant = 0.0f, .linear = 0.09f, .quadratic = 1.0f
    });

    // models
    Model sponza = Model({ .path = modelPath });
    Node sponzaNode = Node(&sponza);
    sponzaNode.setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode.setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));

    Model backpack = Model({ .path = BACKPACK_MODEL_PATH, .flipTextures = true });
    Node backpackNode = Node(&backpack);
    backpackNode.setTranslation(glm::vec3(0.5f, 0.1f, -5.0f));
    backpackNode.setScale(glm::vec3(0.25f));

    // load the HDR environment map
    Texture hdrTexture = Texture({
        .internalFormat = GL_RGB16F,
        .format = GL_RGB,
        .type = GL_FLOAT,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .flipped = true,
        .path = hdrImagePath
    });

    // skybox
    CubeMap envCubeMap({ .width = 512, .height = 512, .type = CUBE_MAP_HDR });

    scene.setDirectionalLight(&directionalLight);
    scene.addPointLight(&pointLight1);
    scene.addPointLight(&pointLight2);
    scene.addPointLight(&pointLight3);
    scene.addPointLight(&pointLight4);
    scene.addChildNode(&cubeNodeGold);
    scene.addChildNode(&cubeNodeIron);
    scene.addChildNode(&sphereNodePlastic);
    scene.addChildNode(&sponzaNode);
    scene.addChildNode(&backpackNode);
    scene.addChildNode(&planeNode);

    scene.equirectToCubeMap(envCubeMap, hdrTexture, equirectToCubeMapShader);
    scene.setupIBL(envCubeMap, convolutionShader, prefilterShader, brdfShader);
    scene.setEnvMap(&envCubeMap);

    app.renderer.updateDirLightShadowMap(dirLightShadowsShader, scene, camera);
    app.renderer.updatePointLightShadowMaps(pointLightShadowsShader, scene, camera);

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

    camera.position = glm::vec3(0.0f, 0.0f, 0.0f);
    camera.updateViewMatrix();

    // camera.setProjectionMatrix(glm::radians(120.0f), (float)screenWidth / (float)screenHeight, 0.1f, 1000.0f);

    int i = 0;
    app.onRender([&](double now, double dt) {
        // handle mouse buttons
        auto mouseButtons = window.getMouseButtons();
        window.setMouseCursor(!mouseButtons.LEFT_PRESSED);
        if (mouseButtons.LEFT_PRESSED) {
            static bool firstMouse = true;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;

            auto cursorPos = window.getCursorPos();
            float xpos = static_cast<float>(cursorPos.x);
            float ypos = static_cast<float>(cursorPos.y);

            if (firstMouse) {
                lastX = xpos;
                lastY = ypos;
                firstMouse = false;
            }

            float xoffset = xpos - lastX;
            float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

            lastX = xpos;
            lastY = ypos;

            camera.processMouseMovement(xoffset, yoffset, true);
        }

        // handle keyboard input
        auto keys = window.getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window.close();
        }

        // render all objects in scene
        app.renderer.drawObjects(pbrShader, scene, camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(backgroundShader, scene, camera);

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
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);

        i++;
        if (i == 100) {
            std::cout << "saving" << std::endl;

            app.renderer.gBuffer.colorBuffer.saveTextureToPNG("color.png");
            // app.renderer.gBuffer.depthBuffer.saveDepthToFile("depth1.bin");

            std::ofstream depthFile;
            depthFile.open("data/depth.bin", std::ios::out | std::ios::binary);

            std::ofstream positionsFile;
            positionsFile.open("data/positions.bin", std::ios::out | std::ios::binary);

            std::ofstream objFile;
            objFile.open("mesh.obj");

            std::ofstream mtlFile;
            mtlFile.open("mesh.mtl");

            std::vector<Vertex> vertices;
            vertices.reserve(screenWidth * screenHeight);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
            GLvoid* pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pBuffer) {
                glm::vec4* pVertices = static_cast<glm::vec4*>(pBuffer);

                for (int i = 0; i < screenWidth * screenHeight; ++i) {
                    Vertex vertex;
                    vertex.position.x = pVertices[i].x;
                    vertex.position.y = pVertices[i].y;
                    vertex.position.z = pVertices[i].z;
                    vertices.push_back(vertex);

                    depthFile.write(reinterpret_cast<const char*>(&pVertices[i].w), sizeof(pVertices[i].w));
                }

                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to map vertex buffer into memory" << std::endl;
            }

            std::vector<unsigned int> indices;
            indices.reserve(indexBufferSize);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
            GLvoid* pIndexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pIndexBuffer) {
                unsigned int* pIndices = static_cast<unsigned int*>(pIndexBuffer);

                for (int i = 0; i < indexBufferSize; ++i) {
                    indices.push_back(pIndices[i]);
                }

                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to map index buffer into memory" << std::endl;
            }

            std::vector<glm::vec2> texCoords;
            texCoords.reserve(screenWidth * screenHeight);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, texCoordBuffer);
            GLvoid* pTexCoordBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pTexCoordBuffer) {
                glm::vec2* pTexCoords = static_cast<glm::vec2*>(pTexCoordBuffer);

                for (int i = 0; i < screenWidth * screenHeight; ++i) {
                    texCoords.push_back(pTexCoords[i]);
                }

                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to map tex coord buffer into memory" << std::endl;
            }

            if (objFile.is_open()) {
                objFile << "mtllib " << "mesh.mtl" << std::endl;
                objFile << "usemtl " << "Material1" << std::endl;
                for (int i = 0; i < vertices.size(); i++) {
                    objFile << "v " << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
                    positionsFile.write(reinterpret_cast<const char*>(&vertices[i].position.x), sizeof(vertices[i].position.x));
                    positionsFile.write(reinterpret_cast<const char*>(&vertices[i].position.y), sizeof(vertices[i].position.y));
                    positionsFile.write(reinterpret_cast<const char*>(&vertices[i].position.z), sizeof(vertices[i].position.z));
                    // positionsFile << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
                }
                for (int i = 0; i < texCoords.size(); i++) {
                    objFile << "vt " << texCoords[i].x << " " << texCoords[i].y << std::endl;
                }
                // for (int i = 0; i < indices.size(); i += 3) {
                //     objFile << "f " << indices[i] + 1 << " " << indices[i + 1] + 1 << " " << indices[i + 2] + 1<< std::endl;
                // }
                for (int i = 0; i < indices.size(); i += 3) {
                    objFile << "f " << indices[i + 0] + 1 << "/" << indices[i + 0] + 1 << " " << indices[i + 1] + 1 << "/" << indices[i + 1] + 1 << " " << indices[i + 2] + 1 << "/" << indices[i + 2] + 1 << std::endl;
                }
                objFile.close();
            } else {
                std::cerr << "Failed to open objFile" << std::endl;
            }

            if (mtlFile.is_open()) {
                mtlFile << "newmtl " << "Material1" << std::endl;
                mtlFile << "map_Kd " << "meshColor.png" << std::endl;
            }

            std::cout << "done saving" << std::endl;
            window.close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
