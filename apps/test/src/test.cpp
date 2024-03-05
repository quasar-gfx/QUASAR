#include <iostream>

#include <imgui/imgui.h>

#include <Shader.h>
#include <Texture.h>
#include <Mesh.h>
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

#define GUI_UPDATE_FRAMERATE_INTERVAL 0.1f // seconds

static const char* SkyBoxShaderVertGlsl = R"_(#version 330 core
    layout (location = 0) in vec3 aPos;

    out vec3 TexCoords;

    uniform mat4 projection;
    uniform mat4 view;

    void main() {
        TexCoords = aPos;
        vec4 pos = projection * view * vec4(aPos, 1.0);
        gl_Position = pos.xyww;
    }
)_";

static const char* SkyBoxShaderFragGlsl = R"_(#version 330 core
    out vec4 FragColor;

    in vec3 TexCoords;

    uniform samplerCube skybox;

    void main() {
        vec4 col = texture(skybox, TexCoords);
        FragColor = col;
    }
)_";

void processInput(OpenGLApp* app, Camera* camera, float deltaTime);

const std::string CONTAINER_TEXTURE = "../../assets/textures/container.jpg";
const std::string METAL_TEXTURE = "../../assets/textures/metal.png";
const std::string BACKPACK_MODEL_PATH = "../../assets/models/backpack/backpack.obj";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Test App";

    std::string modelPath = "../../assets/models/sponza/sponza.obj";
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
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            app.config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
    }

    app.init();

    int width, height;
    app.getWindowSize(&width, &height);

    Scene* scene = new Scene();
    Camera* camera = new Camera(width, height);

    app.gui([&app](double now, double dt) {
        static float deltaTimeSum = 0.0f;
        static int sumCount = 0;
        static float frameRateToDisplay = 0.0f;
        static float prevDisplayTime = 0.0f;

        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        if (now - prevDisplayTime > GUI_UPDATE_FRAMERATE_INTERVAL) {
            prevDisplayTime = now;
            if (deltaTimeSum > 0.0f) {
                frameRateToDisplay = sumCount / deltaTimeSum;
                deltaTimeSum = 0.0f; sumCount = 0;
            }
        }
        deltaTimeSum += dt; sumCount++;
        ImGui::Text("Rendering Frame Rate: %.1f FPS", frameRateToDisplay);
        ImGui::End();
    });

    app.onResize([&camera](unsigned int width, unsigned int height) {
        camera->aspect = (float)width / (float)height;
    });

    app.onMouseMove([&app, &camera](double xposIn, double yposIn) {
        static float lastX = app.config.width / 2.0;
        static float lastY = app.config.height / 2.0;

        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

        lastX = xpos;
        lastY = ypos;

        camera->processMouseMovement(xoffset, yoffset);
    });

    app.onMouseScroll([&app, &camera](double xoffset, double yoffset) {
        camera->processMouseScroll(static_cast<float>(yoffset));
    });

    // shaders
    Shader skyboxShader = Shader::createFromData(SkyBoxShaderVertGlsl, SkyBoxShaderFragGlsl);
    skyboxShader.setInt("skybox", 0);
    Shader shader = Shader::createFromFiles("shaders/meshMaterial.vert", "shaders/meshMaterial.frag");
    Shader screenShader = Shader::createFromFiles("shaders/postprocess.vert", "shaders/postprocess.frag");

    // textures
    Texture* cubeTexture = Texture::create(CONTAINER_TEXTURE);
    std::vector<Texture*> cubeTextures;
    cubeTextures.push_back(cubeTexture);

    Texture* floorTexture = Texture::create(METAL_TEXTURE);
    std::vector<Texture*> floorTextures;
    floorTextures.push_back(floorTexture);

    std::vector<Vertex> cubeVertices {
        // Front face
        { {-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Left
        { { 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Right
        { { 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Right
        { { 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Right
        { {-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Left
        { {-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Left

        // Back face
        { { 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Right
        { {-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Left
        { {-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Left
        { {-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Left
        { { 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Right
        { { 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Right

        // Left face
        { {-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },  // Bottom Front
        { {-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },  // Bottom Back
        { {-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f} },  // Top Back
        { {-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f} },  // Top Back
        { {-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, -1.0f} },  // Top Front
        { {-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },  // Bottom Front

        // Right face
        { { 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },  // Bottom Front
        { { 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },  // Bottom Back
        { { 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },  // Top Back
        { { 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },  // Top Back
        { { 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },  // Top Front
        { { 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },  // Bottom Front

        // Top face
        { {-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Left
        { { 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Right
        { { 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Right
        { { 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Right
        { {-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },  // Bottom Left
        { {-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} },  // Top Left

        // Bottom face
        { {-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Left
        { { 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },  // Bottom Right
        { { 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Right
        { { 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Right
        { {-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f} },  // Top Left
        { {-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} }   // Bottom Left
    };
    Mesh* cubeMesh = Mesh::create(cubeVertices, cubeTextures);

    Node* cubeNode1 = Node::create(cubeMesh);
    cubeNode1->setTranslation(glm::vec3(-1.0f, 0.0f, -1.0f));

    Node* cubeNode2 = Node::create(cubeMesh);
    cubeNode1->addChildNode(cubeNode2);
    cubeNode2->setTranslation(glm::vec3(2.5f, 0.0f, 1.0f));

    // lights
    AmbientLight* ambientLight = AmbientLight::create(glm::vec3(0.9f, 0.9f, 0.9f), 0.7f);

    DirectionalLight* directionalLight = DirectionalLight::create(glm::vec3(0.8f, 0.8f, 0.8f), 0.9f);
    directionalLight->setDirection(glm::vec3(-0.2f, -1.0f, -0.3f));

    PointLight* pointLight1 = PointLight::create(glm::vec3(0.9f, 0.9f, 1.0f), 0.3f);
    pointLight1->setPosition(glm::vec3(-1.45f, 0.9f, -6.2f));
    pointLight1->setAttenuation(1.0f, 0.09f, 0.032f);

    PointLight* pointLight2 = PointLight::create(glm::vec3(0.9f, 0.9f, 1.0f), 0.3f);
    pointLight2->setPosition(glm::vec3(2.2f, 0.9f, -6.2f));
    pointLight2->setAttenuation(1.0f, 0.09f, 0.032f);

    PointLight* pointLight3 = PointLight::create(glm::vec3(0.9f, 0.9f, 1.0f), 0.3f);
    pointLight3->setPosition(glm::vec3(-1.45f, 0.9f, 4.89f));
    pointLight3->setAttenuation(1.0f, 0.09f, 0.032f);

    PointLight* pointLight4 = PointLight::create(glm::vec3(0.9f, 0.9f, 1.0f), 0.3f);
    pointLight4->setPosition(glm::vec3(2.2f, 0.9f, 4.89f));
    pointLight4->setAttenuation(1.0f, 0.09f, 0.032f);

    // models
    Model* sponza = Model::create(modelPath);

    Node* sponzaNode = Node::create(sponza);
    sponzaNode->setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode->setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));
    sponzaNode->setScale(glm::vec3(0.01f));

    Model* backpack = Model::create(BACKPACK_MODEL_PATH, true);

    Node* backpackNode = Node::create(backpack);
    backpackNode->setTranslation(glm::vec3(-0.25f, 0.25f, -3.0f));
    backpackNode->setScale(glm::vec3(0.25f));

    CubeMap* skybox = CubeMap::create({
        "../../assets/textures/skybox/right.jpg",
        "../../assets/textures/skybox/left.jpg",
        "../../assets/textures/skybox/top.jpg",
        "../../assets/textures/skybox/bottom.jpg",
        "../../assets/textures/skybox/front.jpg",
        "../../assets/textures/skybox/back.jpg"
    });

    scene->setAmbientLight(ambientLight);
    scene->setDirectionalLight(directionalLight);
    scene->addPointLight(pointLight1);
    scene->addPointLight(pointLight2);
    scene->addPointLight(pointLight3);
    scene->addPointLight(pointLight4);
    scene->setSkyBox(skybox);
    scene->addChildNode(cubeNode1);
    scene->addChildNode(sponzaNode);
    scene->addChildNode(backpackNode);

    FullScreenQuad* fsQuad = FullScreenQuad::create();

    // framebuffer to render into
    FrameBuffer* framebuffer = FrameBuffer::create(app.config.width, app.config.height);

    app.onRender([&](double now, double dt) {
        processInput(&app, camera, dt);

        // bind to framebuffer and draw scene as we normally would to color texture
        framebuffer->bind();
        glViewport(0, 0, framebuffer->width, framebuffer->height);

        // make sure we clear the framebuffer's content
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // must draw before drawing scene
        app.renderer.drawSkyBox(skyboxShader, scene, camera);

        // draw all objects in scene
        app.renderer.draw(shader, scene, camera);

        // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
        framebuffer->unbind();
        glViewport(0, 0, width, height);

        // clear all relevant buffers
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT);

        screenShader.bind();
        screenShader.setInt("screenTexture", 0);
            framebuffer->bindColorAttachment(0);
                fsQuad->draw();
            framebuffer->unbindColorAttachment();
        screenShader.unbind();
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
