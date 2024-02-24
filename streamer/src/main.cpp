#include <iostream>

#include "Shader.h"
#include "VertexBuffer.h"
#include "Texture.h"
#include "FrameBuffer.h"
#include "OpenGLApp.h"

#include "VideoStreamer.h"

void processInput(OpenGLApp* app);

const std::string CONTAINER_TEXTURE = "../assets/textures/container.jpg";
const std::string METAL_TEXTURE = "../assets/textures/metal.png";

float deltaTime = 0.0f;

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Video Streamer";

    std::string inputFileName = "input.mp4";
    std::string outputUrl = "udp://localhost:1234";
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
            inputFileName = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputUrl = argv[i + 1];
            i++;
        }
    }

    app.init();

    app.mouseMove([&app](double xposIn, double yposIn) {
        static float lastX = app.config.width / 2.0;
        static float lastY = app.config.height / 2.0;

        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

        lastX = xpos;
        lastY = ypos;

        app.camera.processMouseMovement(xoffset, yoffset);
    });

    app.mouseScroll([&app](double xoffset, double yoffset) {
        app.camera.processMouseScroll(static_cast<float>(yoffset));
    });

    Shader shader("shaders/framebuffer.vert", "shaders/framebuffer.frag");
    Shader screenShader("shaders/postprocess.vert", "shaders/postprocess.frag");

    float cubeVertices[] = {
        // positions          // texture Coords
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    float planeVertices[] = {
        // positions          // texture Coords
         5.0f, -0.5f,  5.0f,  2.0f, 0.0f,
        -5.0f, -0.5f,  5.0f,  0.0f, 0.0f,
        -5.0f, -0.5f, -5.0f,  0.0f, 2.0f,

         5.0f, -0.5f,  5.0f,  2.0f, 0.0f,
        -5.0f, -0.5f, -5.0f,  0.0f, 2.0f,
         5.0f, -0.5f, -5.0f,  2.0f, 2.0f
    };
    float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    // cube vertices
    VertexBuffer cubeVB(cubeVertices, sizeof(cubeVertices));
    cubeVB.bind();
    cubeVB.addAttribute(0, 3, GL_FALSE, 5 * sizeof(float), 0);
    cubeVB.addAttribute(1, 2, GL_FALSE, 5 * sizeof(float), 3 * sizeof(float));

    // plane vertices
    VertexBuffer planeVB(planeVertices, sizeof(planeVertices));
    planeVB.bind();
    planeVB.addAttribute(0, 3, GL_FALSE, 5 * sizeof(float), 0);
    planeVB.addAttribute(1, 2, GL_FALSE, 5 * sizeof(float), 3 * sizeof(float));

    // screen quad vertices
    VertexBuffer quadVB(quadVertices, sizeof(quadVertices));
    quadVB.bind();
    quadVB.addAttribute(0, 2, GL_FALSE, 4 * sizeof(float), 0);
    quadVB.addAttribute(1, 2, GL_FALSE, 4 * sizeof(float), 2 * sizeof(float));

    // textures
    Texture cubeTexture(CONTAINER_TEXTURE);
    Texture floorTexture(METAL_TEXTURE);

    // shaders
    shader.bind();
    shader.setInt("texture1", 0);

    screenShader.bind();
    screenShader.setInt("screenTexture", 0);

    VideoStreamer videoStreamer{};
    int ret = videoStreamer.init(inputFileName, outputUrl);
    if (ret < 0) {
        std::cerr << "Failed to initialize FFMpeg Video Streamer" << std::endl;
        return ret;
    }

    // framebuffer to render into
    FrameBuffer framebuffer(app.config.width, app.config.height);

    app.animate([&](double now, double dt) {
        deltaTime = dt;

        processInput(&app);

        // bind to framebuffer and draw scene as we normally would to color texture
        framebuffer.bind();

            glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)

            // make sure we clear the framebuffer's content
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            shader.bind();
            shader.setMat4("view", app.camera.view);
            shader.setMat4("projection", app.camera.proj);

            glm::mat4 model;

            // cubes
            cubeTexture.bind();
            cubeVB.bind();
                // cube 1
                model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(-1.0f, 0.0f, -1.0f));
                shader.setMat4("model", model);
                glDrawArrays(GL_TRIANGLES, 0, 36);

                // cube 2
                model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(2.0f, 0.0f, 0.0f));
                shader.setMat4("model", model);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            cubeVB.unbind();

            // floor
            floorTexture.bind();
            planeVB.bind();
                model = glm::mat4(1.0f);
                shader.setMat4("model", model);
                glDrawArrays(GL_TRIANGLES, 0, 6);
            planeVB.unbind();

        // now bind back to default framebuffer and draw a quad plane with the attached framebuffer color texture
        framebuffer.unbind();

        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
        // clear all relevant buffers
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
        glClear(GL_COLOR_BUFFER_BIT);

        screenShader.bind();

        framebuffer.colorAttachment.bind();
        quadVB.bind();
            glDrawArrays(GL_TRIANGLES, 0, 6);
        quadVB.unbind();

        // @TODO make this stream the framebuffer to the output URL
        ret = videoStreamer.sendFrame();
        if (ret < 0) {
            return;
        }
    });

    // run app loop (blocking)
    app.run();

    // cleanup
    cubeVB.cleanup();
    planeVB.cleanup();
    quadVB.cleanup();
    cubeTexture.cleanup();
    floorTexture.cleanup();
    framebuffer.cleanup();
    videoStreamer.cleanup();
    app.cleanup();

    return 0;
}

void processInput(OpenGLApp* app) {
    if (glfwGetKey(app->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(app->window, true);

    if (glfwGetKey(app->window, GLFW_KEY_W) == GLFW_PRESS)
        app->camera.processKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_S) == GLFW_PRESS)
        app->camera.processKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_A) == GLFW_PRESS)
        app->camera.processKeyboard(LEFT, deltaTime);
    if (glfwGetKey(app->window, GLFW_KEY_D) == GLFW_PRESS)
        app->camera.processKeyboard(RIGHT, deltaTime);
}
