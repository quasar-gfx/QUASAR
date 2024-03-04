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
#include <OpenGLApp.h>

#define GUI_UPDATE_FRAMERATE_INTERVAL 0.1f // seconds

void processInput(OpenGLApp* app, Camera* camera, float deltaTime);

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Test App";

    std::string modelPath = "../../assets/models/cerberus/cerberus.fbx";
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
    Shader skyboxShader = Shader::createFromFiles("shaders/skybox.vert", "shaders/skybox.frag");
    skyboxShader.setInt("skybox", 0);
    Shader shader = Shader::createFromFiles("shaders/meshMaterial.vert", "shaders/meshMaterial.frag");
    Shader screenShader = Shader::createFromFiles("shaders/postprocess.vert", "shaders/postprocess.frag");

    // lights
    AmbientLight* ambientLight = AmbientLight::create(glm::vec3(0.9f, 0.9f, 0.9f), 0.7f);

    DirectionalLight* directionalLight = DirectionalLight::create(glm::vec3(0.8f, 0.8f, 0.8f), 0.9f);
    directionalLight->setDirection(glm::vec3(-0.2f, -1.0f, -0.3f));

    PointLight* pointLight = PointLight::create(glm::vec3(0.9f, 0.9f, 1.0f), 0.6f);
    pointLight->setPosition(glm::vec3(0.0f, 3.0f, 0.0f));
    pointLight->setAttenuation(1.0f, 0.09f, 0.032f);

    // models
    Model* gun = Model::create(modelPath);

    Node* gunNode = Node::create(gun);
    gunNode->setTranslation(glm::vec3(0.0f, 1.0f, -1.0f));
    gunNode->setRotationEuler(glm::vec3(-90.0f, 90.0f, 0.0f));
    gunNode->setScale(glm::vec3(0.05f));

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
    scene->addPointLight(pointLight);
    scene->setSkyBox(skybox);
    scene->addChildNode(gunNode);

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
        app.drawSkyBox(skyboxShader, scene, camera);

        // draw all objects in scene
        app.draw(shader, scene, camera);

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
