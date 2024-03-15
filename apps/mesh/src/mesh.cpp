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
#include <FrameBuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>

#include <glm/gtx/string_cast.hpp>

void processInput(OpenGLApp* app, Camera* camera, float deltaTime);

const std::string BACKPACK_MODEL_PATH = "../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Mesh Test";
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

    app.init();

    int screenWidth, screenHeight;
    app.getWindowSize(&screenWidth, &screenHeight);

    Scene* scene = new Scene();
    Camera* camera = new Camera(screenWidth, screenHeight);

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();
    });

    app.onResize([&camera](unsigned int width, unsigned int height) {
        camera->aspect = (float)width / (float)height;
    });

    app.onMouseMove([&app, &camera](double xposIn, double yposIn) {
        static bool mouseDown = false;

        static float lastX = app.config.width / 2.0;
        static float lastY = app.config.height / 2.0;

        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

        lastX = xpos;
        lastY = ypos;

        if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            mouseDown = true;
            glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }

        if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            mouseDown = false;
            glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }

        if (mouseDown) {
            camera->processMouseMovement(xoffset, yoffset);
        }
    });

    app.onMouseScroll([&app, &camera](double xoffset, double yoffset) {
        camera->processMouseScroll(static_cast<float>(yoffset));
    });

    // shaders
    Shader pbrShader, screenShader;
    pbrShader.loadFromFile("../assets/shaders/pbr/pbr.vert", "../assets/shaders/pbr/pbr.frag");
    screenShader.loadFromFile("../assets/shaders/postprocessing/postprocess.vert", "../assets/shaders/postprocessing/displayDepth.frag");

    // converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader;
    equirectToCubeMapShader.loadFromFile("../assets/shaders/cubemap/cubemap.vert", "../assets/shaders/cubemap/equirectangular2cubemap.frag");

    // solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader;
    convolutionShader.loadFromFile("../assets/shaders/cubemap/cubemap.vert", "../assets/shaders/pbr/irradianceConvolution.frag");

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader;
    prefilterShader.loadFromFile("../assets/shaders/cubemap/cubemap.vert", "../assets/shaders/pbr/prefilter.frag");

    // BRDF shader
    Shader brdfShader;
    brdfShader.loadFromFile("../assets/shaders/pbr/brdf.vert", "../assets/shaders/pbr/brdf.frag");

    // background skybox shader
    Shader backgroundShader;
    backgroundShader.loadFromFile("../assets/shaders/cubemap/background.vert", "../assets/shaders/cubemap/backgroundHDR.frag");

    // textures
    Texture albedo = Texture("../assets/textures/pbr/gold/albedo.png");
    Texture normal = Texture("../assets/textures/pbr/gold/normal.png");
    Texture metallic = Texture("../assets/textures/pbr/gold/metallic.png");
    Texture roughness = Texture("../assets/textures/pbr/gold/roughness.png");
    Texture ao = Texture("../assets/textures/pbr/gold/ao.png");
    std::vector<TextureID> goldTextures = { albedo.ID, 0, normal.ID, metallic.ID, roughness.ID, ao.ID };

    Texture ironAlbedo = Texture("../assets/textures/pbr/rusted_iron/albedo.png");
    Texture ironNormal = Texture("../assets/textures/pbr/rusted_iron/normal.png");
    Texture ironMetallic = Texture("../assets/textures/pbr/rusted_iron/metallic.png");
    Texture ironRoughness = Texture("../assets/textures/pbr/rusted_iron/roughness.png");
    Texture ironAo = Texture("../assets/textures/pbr/rusted_iron/ao.png");
    std::vector<TextureID> ironTextures = { ironAlbedo.ID, 0, ironNormal.ID, ironMetallic.ID, ironRoughness.ID, ironAo.ID };

    Texture plasticAlbedo = Texture("../assets/textures/pbr/plastic/albedo.png");
    Texture plasticNormal = Texture("../assets/textures/pbr/plastic/normal.png");
    Texture plasticMetallic = Texture("../assets/textures/pbr/plastic/metallic.png");
    Texture plasticRoughness = Texture("../assets/textures/pbr/plastic/roughness.png");
    Texture plasticAo = Texture("../assets/textures/pbr/plastic/ao.png");
    std::vector<TextureID> plasticTextures = { plasticAlbedo.ID, 0, plasticNormal.ID, plasticMetallic.ID, plasticRoughness.ID, plasticAo.ID };

    Texture windowTexture = Texture("../assets/textures/window.png");
    std::vector<TextureID> windowTextures = { windowTexture.ID };

    // objects
    Cube* cubeGold = new Cube(goldTextures);
    Node* cubeNodeGold = new Node(cubeGold);
    cubeNodeGold->setTranslation(glm::vec3(-0.2f, 0.25f, -7.0f));
    cubeNodeGold->setScale(glm::vec3(0.5f));

    Cube* cubeIron = new Cube(ironTextures);
    Node* cubeNodeIron = new Node(cubeIron);
    cubeNodeIron->setTranslation(glm::vec3(1.5f, 0.25f, -3.0f));
    cubeNodeIron->setScale(glm::vec3(0.5f));

    Sphere* sphere = new Sphere(plasticTextures);
    Node* sphereNodePlastic = new Node(sphere);
    sphereNodePlastic->setTranslation(glm::vec3(1.0f, 1.5f, -8.0f));
    sphereNodePlastic->setScale(glm::vec3(0.5f));

    Plane* plane = new Plane(windowTextures);
    Node* planeNode = new Node(plane);
    planeNode->setTranslation(glm::vec3(0.0f, 1.5f, -7.0f));
    planeNode->setRotationEuler(glm::vec3(-90.0f, 0.0f, 0.0f));
    planeNode->setScale(glm::vec3(0.5f));

    // lights
    DirectionalLight* directionalLight = new DirectionalLight(glm::vec3(0.8f, 0.8f, 0.8f), 0.1f);
    directionalLight->setDirection(glm::vec3(0.0f, -1.0f, -0.3f));

    PointLight* pointLight1 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight1->setPosition(glm::vec3(-1.45f, 3.5f, -6.2f));
    pointLight1->setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight* pointLight2 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight2->setPosition(glm::vec3(2.2f, 3.5f, -6.2f));
    pointLight2->setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight* pointLight3 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight3->setPosition(glm::vec3(-1.45f, 3.5f, 4.89f));
    pointLight3->setAttenuation(0.0f, 0.09f, 1.0f);

    PointLight* pointLight4 = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 100.0f);
    pointLight4->setPosition(glm::vec3(2.2f, 3.5f, 4.89f));
    pointLight4->setAttenuation(0.0f, 0.09f, 1.0f);

    // models
    Model* sponza = new Model(modelPath);

    Node* sponzaNode = new Node(sponza);
    sponzaNode->setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode->setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));

    Model* backpack = new Model(BACKPACK_MODEL_PATH, true);

    Node* backpackNode = new Node(backpack);
    backpackNode->setTranslation(glm::vec3(0.5f, 0.1f, -5.0f));
    backpackNode->setScale(glm::vec3(0.25f));

    // load the HDR environment map
    Texture hdrTexture(hdrImagePath, GL_FLOAT, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR, true);

    // skybox
    CubeMap envCubeMap(512, 512, CUBE_MAP_HDR);

    scene->setDirectionalLight(directionalLight);
    scene->addPointLight(pointLight1);
    scene->addPointLight(pointLight2);
    scene->addPointLight(pointLight3);
    scene->addPointLight(pointLight4);
    scene->addChildNode(cubeNodeGold);
    scene->addChildNode(cubeNodeIron);
    scene->addChildNode(sphereNodePlastic);
    // scene->addChildNode(sponzaNode);
    scene->addChildNode(backpackNode);
    scene->addChildNode(planeNode);

    scene->equirectToCubeMap(envCubeMap, hdrTexture, equirectToCubeMapShader);
    scene->setupIBL(envCubeMap, convolutionShader, prefilterShader, brdfShader);
    scene->setEnvMap(&envCubeMap);

    Shader dirLightShadowsShader;
    dirLightShadowsShader.loadFromFile("../assets/shaders/shadows/dirShadow.vert", "../assets/shaders/shadows/dirShadow.frag");

    Shader pointLightShadowsShader;
    pointLightShadowsShader.loadFromFile("../assets/shaders/shadows/pointShadow.vert", "../assets/shaders/shadows/pointShadow.frag", "../assets/shaders/shadows/pointShadow.geo");

    ComputeShader genMeshShader;
    genMeshShader.loadFromFile("../assets/shaders/compute/genMesh.comp");

    app.renderer.updateDirLightShadowMap(dirLightShadowsShader, scene, camera);
    app.renderer.updatePointLightShadowMaps(pointLightShadowsShader, scene, camera);

    GLuint vertexBuffer, indexBuffer;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, screenWidth * screenHeight * sizeof(glm::vec4), nullptr, GL_STATIC_DRAW);

    // Generate index buffer
    int numTriangles = (screenWidth - 1) * (screenHeight - 1) * 2;
    int indexBufferSize = numTriangles * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    genMeshShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    genMeshShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genMeshShader.unbind();

    int i = 0;

    app.onRender([&](double now, double dt) {
        processInput(&app, camera, dt);

        // render all objects in scene
        app.renderer.drawObjects(pbrShader, scene, camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(backgroundShader, scene, camera);

        genMeshShader.bind();
        genMeshShader.setMat4("view", camera->getViewMatrix());
        genMeshShader.setMat4("projectionInverse", glm::inverse(camera->getProjectionMatrix()));
        genMeshShader.setFloat("near", camera->near);
        genMeshShader.setFloat("far", camera->far);
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
                    std::cout << pVertices[i].x << " " << pVertices[i].y << " " << pVertices[i].z << " " << pVertices[i].w << std::endl;
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

            std::cout << "saving" << std::endl;
            std::ofstream file;
            file.open("mesh.obj");
            if (file.is_open()) {
                for (int i = 0; i < vertices.size(); i++) {
                    file << "v " << vertices[i].position.x << " " << vertices[i].position.y << " " << vertices[i].position.z << std::endl;
                }
                for (int i = 0; i < indices.size(); i += 3) {
                    file << "f " << indices[i] + 1 << " " << indices[i + 1] + 1 << " " << indices[i + 2] + 1<< std::endl;
                }
                file.close();
            } else {
                std::cerr << "Failed to open file" << std::endl;
            }
            std::cout << "done saving" << std::endl;
        }
    });

    // run app loop (blocking)
    app.run();

    // cleanup
    app.cleanup();

    return 0;
}

void processInput(OpenGLApp* app, Camera* camera, float deltaTime) {
    if (glfwGetKey(app->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(app->window, true);

    if (glfwGetKey(app->window, GLFW_KEY_W) == GLFW_PRESS)
        camera->processKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_S) == GLFW_PRESS)
        camera->processKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_A) == GLFW_PRESS)
        camera->processKeyboard(LEFT, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_D) == GLFW_PRESS)
        camera->processKeyboard(RIGHT, deltaTime);
}
