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
#include <FrameBuffer.h>
#include <FullScreenQuad.h>
#include <OpenGLApp.h>

#include <cuda.h>

#include <VideoStreamer.h>

#define GUI_UPDATE_FRAMERATE_INTERVAL 0.1f // seconds

void processInput(OpenGLApp* app, Camera* camera, float deltaTime);

int getDeviceName(std::string& gpuName) {
    // Setup the cuda context for hardware encoding with ffmpeg
    int iGpu = 0;
    CUresult res;
    cuInit(0);
    int nGpu = 0;
    cuDeviceGetCount(&nGpu);
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        return 1;
    }

    CUdevice cuDevice = 0;
    cuDeviceGet(&cuDevice, iGpu);
    char szDeviceName[80];
    cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
    gpuName = szDeviceName;

    return 0;
}

const std::string CONTAINER_TEXTURE = "../../assets/textures/container.jpg";
const std::string METAL_TEXTURE = "../../assets/textures/metal.png";

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Video Streamer";

    std::string inputFileName = "input.mp4";
    std::string outputUrl = "udp://localhost:1234";
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
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            inputFileName = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputUrl = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) {
            app.config.enableVSync = atoi(argv[i + 1]);
            i++;
        }
    }

    std::string gpuName;
    if (getDeviceName(gpuName) != 0) {
        return 1;
    }
    std::cout << "GPU in use: " << gpuName << std::endl;

    app.init();

    int width, height;
    app.getWindowSize(&width, &height);

    Scene* scene = new Scene();
    Camera* camera = new Camera(width, height);

    VideoStreamer* videoStreamer = VideoStreamer::create();

    app.gui([&app, &videoStreamer](double now, double dt) {
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
        ImGui::Text("Video Frame Rate: %.1f FPS", videoStreamer->getFrameRate());
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
    Shader skyboxShader("shaders/skybox.vert", "shaders/skybox.frag");
    skyboxShader.setInt("skybox", 0);
    Shader shader("shaders/framebuffer.vert", "shaders/framebuffer.frag");
    Shader screenShader("shaders/postprocess.vert", "shaders/postprocess.frag");

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
    cubeNode2->setTranslation(glm::vec3(2.0f, 0.0f, 0.0f));

    Model* sponza = Model::create(modelPath);

    Node* sponzaNode = Node::create(sponza);
    sponzaNode->setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
    sponzaNode->setRotationEuler(glm::vec3(0.0f, -90.0f, 0.0f));
    sponzaNode->setScale(glm::vec3(0.01f));

    scene->addChildNode(cubeNode1);
    scene->addChildNode(cubeNode2);
    scene->addChildNode(sponzaNode);

    int ret = videoStreamer->start(inputFileName, outputUrl);
    if (ret < 0) {
        std::cerr << "Failed to initialize FFMpeg Video Streamer" << std::endl;
        return ret;
    }

    CubeMap* skybox = CubeMap::create({
        "../../assets/textures/skybox/right.jpg",
        "../../assets/textures/skybox/left.jpg",
        "../../assets/textures/skybox/top.jpg",
        "../../assets/textures/skybox/bottom.jpg",
        "../../assets/textures/skybox/front.jpg",
        "../../assets/textures/skybox/back.jpg"
    });

    FullScreenQuad* fsQuad = FullScreenQuad::create();

    // framebuffer to render into
    FrameBuffer* framebuffer = FrameBuffer::create(width, height);

    // enable backface culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    // enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    app.onRender([&](double now, double dt) {
        processInput(&app, camera, dt);

        // bind to framebuffer and draw scene as we normally would to color texture
        framebuffer->bind();

        // make sure we clear the framebuffer's content
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // must draw before drawing scene
        skybox->draw(skyboxShader, camera);

        // draw all objects in scene
        app.draw(shader, scene, camera);

        // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
        framebuffer->unbind();

        // clear all relevant buffers
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT);

        screenShader.bind();
        screenShader.setInt("screenTexture", 0);
            framebuffer->bindColorAttachment();
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
