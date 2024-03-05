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
    Shader shader, backgroundShader, screenShader;
    shader.loadFromFile("shaders/meshMaterial.vert", "shaders/meshMaterial.frag");
    backgroundShader.loadFromFile("shaders/background.vert", "shaders/background.frag");
    backgroundShader.setInt("environmentMap", 0);
    screenShader.loadFromFile("shaders/postprocess.vert", "shaders/postprocess.frag");

    // lights
    AmbientLight* ambientLight = new AmbientLight(glm::vec3(0.9f, 0.9f, 0.9f), 0.7f);

    DirectionalLight* directionalLight = new DirectionalLight(glm::vec3(0.8f, 0.8f, 0.8f), 0.9f);
    directionalLight->setDirection(glm::vec3(-0.2f, -1.0f, -0.3f));

    PointLight* pointLight = new PointLight(glm::vec3(0.9f, 0.9f, 1.0f), 0.6f);
    pointLight->setPosition(glm::vec3(0.0f, 3.0f, 0.0f));
    pointLight->setAttenuation(1.0f, 0.09f, 0.032f);

    // set up framebuffer
    FrameBuffer captureFramebuffer = FrameBuffer(512, 512);

    // load the HDR environment map
    Texture hdrTexture = Texture("../../assets/textures/environment.hdr", GL_FLOAT, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR);

    // convert HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader;
    equirectToCubeMapShader.loadFromFile("shaders/skybox.vert", "shaders/equirectangular2cubemap.frag");

    CubeMap envCubeMap = CubeMap(512, 512, CUBE_MAP_HDR);

    captureFramebuffer.bind();
    envCubeMap.loadFromEquirectTexture(equirectToCubeMapShader, 512, 512, hdrTexture);
    captureFramebuffer.unbind();

    // glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

    // create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap = CubeMap(32, 32, CUBE_MAP_HDR);

    // solve diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader;
    convolutionShader.loadFromFile("shaders/skybox.vert", "shaders/irradianceConvolution.frag");

    captureFramebuffer.bind();
    irradianceCubeMap.convolve(convolutionShader, 32, 32, envCubeMap.ID);
    captureFramebuffer.unbind();

    // create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap = CubeMap(128, 128, CUBE_MAP_PREFILTER);

    // run a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader;
    prefilterShader.loadFromFile("shaders/skybox.vert", "shaders/prefilter.frag");

    captureFramebuffer.bind();
    prefilterCubeMap.prefilter(prefilterShader, 32, 32, envCubeMap.ID, captureFramebuffer.depthAttachment);
    captureFramebuffer.unbind();

    // generate a 2D LUT from the BRDF equations used
    Texture brdfLUT = Texture(512, 512, GL_RG16F, GL_RG, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR);

    // then reconfigure capture framebuffer object and render screen-space quad with BRDF shader
    Shader brdfShader;
    brdfShader.loadFromFile("shaders/brdf.vert", "shaders/brdf.frag");

    captureFramebuffer.bind();
    captureFramebuffer.bindColorAttachment(0);
    captureFramebuffer.bindDepthAttachment(0);

    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

    FullScreenQuad brdfFsQuad = FullScreenQuad();

    glViewport(0, 0, 512, 512);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    brdfShader.bind();
    brdfFsQuad.draw();
    brdfShader.unbind();

    captureFramebuffer.unbindColorAttachment();
    captureFramebuffer.unbindDepthAttachment();
    captureFramebuffer.unbind();

    // models
    Model* gun = new Model(modelPath);

    Node* gunNode = new Node(gun);
    gunNode->setTranslation(glm::vec3(0.0f, 1.0f, -1.0f));
    gunNode->setRotationEuler(glm::vec3(-90.0f, 90.0f, 0.0f));
    gunNode->setScale(glm::vec3(0.05f));

    scene->setAmbientLight(ambientLight);
    scene->setDirectionalLight(directionalLight);
    scene->setSkyBox(&envCubeMap);
    scene->addPointLight(pointLight);
    scene->addChildNode(gunNode);

    FullScreenQuad fsQuad = FullScreenQuad();

    // framebuffer to render into
    FrameBuffer framebuffer = FrameBuffer(app.config.width, app.config.height);

    app.onRender([&](double now, double dt) {
        processInput(&app, camera, dt);

        // bind to framebuffer and draw scene as we normally would to color texture
        framebuffer.bind();
        glViewport(0, 0, framebuffer.width, framebuffer.height);

        // make sure we clear the framebuffer's content
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // draw all objects in scene
        app.renderer.draw(shader, scene, camera);

        // render skybox (render as last to prevent overdraw)
        app.renderer.drawSkyBox(backgroundShader, scene, camera);

        // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
        framebuffer.unbind();
        glViewport(0, 0, width, height);

        // clear all relevant buffers
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT);

        screenShader.bind();
        screenShader.setInt("screenTexture", 0);
            framebuffer.bindColorAttachment(0);
                fsQuad.draw();
            framebuffer.unbindColorAttachment();
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
