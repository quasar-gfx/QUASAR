#include <iostream>

#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Materials/PBRMaterial.h>
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

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Test App";

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
    SceneLoader loader = SceneLoader();
    bool res = loader.loadScene(scenePath, scene, camera);
    if (!res) {
        std::cerr << "Failed to load scene: " << scenePath << std::endl;
        return 1;
    }

    float exposure = 1.0f;
    int shaderIndex = 0;
    app.gui([&](double now, double dt) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin(app.config.title.c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::TextColored(ImVec4(1,1,0,1), "OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::TextColored(ImVec4(1,1,0,1), "GPU: %s\n", glGetString(GL_RENDERER));

        ImGui::Text("Rendering Frame Rate: %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);

        ImGui::InputFloat3("Camera Position", (float*)&camera.position);
        ImGui::SliderFloat("Movement speed", &camera.movementSpeed, 0.1f, 20.0f);

        if (ImGui::CollapsingHeader("Directional Light Settings")) {
            ImGui::TextColored(ImVec4(1,1,1,1), "Directional Light Settings");
            ImGui::ColorEdit3("Color", (float *)&scene.directionalLight->color);
            ImGui::SliderFloat("Strength", &scene.directionalLight->intensity, 0.1f, 100.0f);
            ImGui::SliderFloat3("Direction", (float*)&scene.directionalLight->direction, -5.0f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Post Processing Settings")) {
            ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f);
            ImGui::RadioButton("Show Color", &shaderIndex, 0);
            ImGui::RadioButton("Show Depth", &shaderIndex, 1);
            ImGui::RadioButton("Show Positions", &shaderIndex, 2);
            ImGui::RadioButton("Show Normals", &shaderIndex, 3);
        }

        ImGui::End();
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    // shaders
    Shader showColorShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader showDepthShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayDepth.frag"
    });

    Shader showPositionShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayPositions.frag"
    });

    Shader showNormalShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayNormals.frag"
    });

    app.onRender([&](double now, double dt) {
        // handle mouse input
        ImGuiIO& io = ImGui::GetIO();
        if (!(io.WantCaptureKeyboard || io.WantCaptureMouse)) {
            auto mouseButtons = window.getMouseButtons();
            window.setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;
            if (!prevMouseLeftPressed && mouseButtons.LEFT_PRESSED) {
                dragging = true;
                prevMouseLeftPressed = true;

                auto cursorPos = window.getCursorPos();
                lastX = static_cast<float>(cursorPos.x);
                lastY = static_cast<float>(cursorPos.y);
            }
            if (prevMouseLeftPressed && !mouseButtons.LEFT_PRESSED) {
                dragging = false;
                prevMouseLeftPressed = false;
            }
            if (dragging) {
                auto cursorPos = window.getCursorPos();
                float xpos = static_cast<float>(cursorPos.x);
                float ypos = static_cast<float>(cursorPos.y);

                float xoffset = xpos - lastX;
                float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

                lastX = xpos;
                lastY = ypos;

                camera.processMouseMovement(xoffset, yoffset, true);
            }
        }

        // handle keyboard input
        auto keys = window.getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window.close();
        }

        // render all objects in scene
        app.renderer.drawObjects(scene, camera);

        // render to screen
        if (shaderIndex == 1) {
            showDepthShader.bind();
            showDepthShader.setFloat("near", camera.near);
            showDepthShader.setFloat("far", camera.far);
            app.renderer.drawToScreen(showDepthShader);
        }
        else if (shaderIndex == 2) {
            showPositionShader.bind();
            app.renderer.drawToScreen(showPositionShader);
        }
        else if (shaderIndex == 3) {
            showNormalShader.bind();
            app.renderer.drawToScreen(showNormalShader);
        }
        else {
            showColorShader.bind();
            showColorShader.setFloat("exposure", exposure);
            app.renderer.drawToScreen(showColorShader);
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
