#include <iostream>
#include <thread>

#include <imgui/imgui.h>

#include <Shader.h>
#include <Texture.h>
#include <Mesh.h>
#include <CubeMap.h>
#include <Entity.h>
#include <Scene.h>
#include <Camera.h>
#include <FrameBuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>

#include <VideoTexture.h>

void processInput(OpenGLApp* app, Camera* camera, float deltaTime);

const std::string CONTAINER_TEXTURE = "../assets/textures/container.jpg";
const std::string METAL_TEXTURE = "../assets/textures/metal.png";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Video Receiver";

    std::string inputUrl = "udp://localhost:1234";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            app.config.width = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) {
            app.config.height = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            inputUrl = argv[i + 1];
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

    VideoTexture videoTexture(app.config.width, app.config.height);
    videoTexture.initVideo(inputUrl);

    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Text("Video Frame Rate: %.1f FPS (%.3f ms/frame)", videoTexture.getFrameRate(), 1000.0f / videoTexture.getFrameRate());
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
    Shader skyboxShader, shader, screenShader;
    skyboxShader.loadFromFile("../assets/shaders/cubemap/background.vert", "../assets/shaders/cubemap/backgroundNoHDR.frag");
    shader.loadFromFile("../assets/shaders/simple.vert", "../assets/shaders/simple.frag");
    screenShader.loadFromFile("../assets/shaders/postprocessing/postprocess.vert", "../assets/shaders/postprocessing/displayVideo.frag");

    // lights
    AmbientLight* ambientLight = new AmbientLight();

    // textures
    Texture cubeTexture = Texture(CONTAINER_TEXTURE);
    std::vector<TextureID> cubeTextures = { cubeTexture.ID };

    Texture floorTexture = Texture(METAL_TEXTURE);
    std::vector<TextureID> floorTextures = { floorTexture.ID };

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
    Mesh* cubeMesh = new Mesh(cubeVertices, cubeTextures);

    Node* cubeNode1 = new Node(cubeMesh);
    cubeNode1->setTranslation(glm::vec3(-1.0f, 0.0f, -1.0f));

    Node* cubeNode2 = new Node(cubeMesh);
    cubeNode2->setTranslation(glm::vec3(2.0f, 0.0f, 0.0f));

    std::vector<Vertex> planeVertices = {
        {{ 25.0f, -0.5f, -25.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 2.0f}, {1.0f, 0.0f, 0.0f}},
        {{-25.0f, -0.5f, -25.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 2.0f}, {1.0f, 0.0f, 0.0f}},
        {{-25.0f, -0.5f,  25.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},

        {{ 25.0f, -0.5f, -25.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 2.0f}, {1.0f, 0.0f, 0.0f}},
        {{-25.0f, -0.5f,  25.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{ 25.0f, -0.5f,  25.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}
    };
    Mesh* planeMesh = new Mesh(planeVertices, floorTextures);

    Node* planeNode = new Node(planeMesh);

    CubeMap* skybox = new CubeMap({
        "../assets/textures/skybox/right.jpg",
        "../assets/textures/skybox/left.jpg",
        "../assets/textures/skybox/top.jpg",
        "../assets/textures/skybox/bottom.jpg",
        "../assets/textures/skybox/front.jpg",
        "../assets/textures/skybox/back.jpg"
    });

    scene->setAmbientLight(ambientLight);
    scene->setEnvMap(skybox);
    scene->addChildNode(cubeNode1);
    scene->addChildNode(cubeNode2);
    scene->addChildNode(planeNode);

    app.onRender([&](double now, double dt) {
        processInput(&app, camera, dt);

        // render all objects in scene
        app.renderer.drawObjects(shader, scene, camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(skyboxShader, scene, camera);

        // render video
        screenShader.bind();
        screenShader.setInt("videoTexture", 4);
        videoTexture.bind(4);
        videoTexture.draw();

        // render to screen
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);
    });

    // run app loop (blocking)
    app.run();

    // cleanup
    app.cleanup();

    std::cout << "Please do CTRL-C to exit!" << std::endl;

    videoTexture.cleanup();

    return 0;
}

void processInput(OpenGLApp* app, Camera* camera,  float deltaTime) {
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
