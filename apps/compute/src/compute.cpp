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

int main(int argc, char** argv) {
    OpenGLApp app{};
    app.config.title = "Compute Shader Test";
    app.config.openglMajorVersion = 4;
    app.config.openglMinorVersion = 3;
    app.config.width = 1000;
    app.config.height = 1000;
    app.config.enableVSync = false;

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

    ComputeShader computeShader;
    computeShader.loadFromFile("../assets/shaders/compute/test.comp");

    Shader screenShader;
    screenShader.loadFromFile("../assets/shaders/postprocessing/postprocess.vert", "../assets/shaders/postprocessing/displayTexture.frag");

    Texture outputTexture = Texture({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_RGBA32F,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST
    });

    // query limitations
	int max_compute_work_group_count[3];
	int max_compute_work_group_size[3];
	int max_compute_work_group_invocations;

	for (int idx = 0; idx < 3; idx++) {
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, idx, &max_compute_work_group_count[idx]);
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, idx, &max_compute_work_group_size[idx]);
	}
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_compute_work_group_invocations);

	std::cout << "OpenGL Limitations: " << std::endl;
	std::cout << "maximum number of work groups in X dimension " << max_compute_work_group_count[0] << std::endl;
	std::cout << "maximum number of work groups in Y dimension " << max_compute_work_group_count[1] << std::endl;
	std::cout << "maximum number of work groups in Z dimension " << max_compute_work_group_count[2] << std::endl;

	std::cout << "maximum size of a work group in X dimension " << max_compute_work_group_size[0] << std::endl;
	std::cout << "maximum size of a work group in Y dimension " << max_compute_work_group_size[1] << std::endl;
	std::cout << "maximum size of a work group in Z dimension " << max_compute_work_group_size[2] << std::endl;

	std::cout << "Number of invocations in a single local work group that may be dispatched to a compute shader " << max_compute_work_group_invocations << std::endl;

    app.onRender([&](double now, double dt) {
        // compute shader
        computeShader.bind();
		computeShader.setFloat("t", now);
	    glBindImageTexture(0, outputTexture.ID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        computeShader.dispatch(screenWidth / 10, screenHeight / 10, 1);
        computeShader.unbind();

        screenShader.bind();
        screenShader.setInt("tex", 4);
        outputTexture.bind(4);

        // render to screen
        app.renderer.drawToScreen(screenShader, screenWidth, screenHeight);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
