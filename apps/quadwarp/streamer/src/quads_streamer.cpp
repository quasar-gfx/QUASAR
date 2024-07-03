#include <iostream>

#include <args.hxx>
#include <imgui/imgui.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <RenderTargets/RenderTarget.h>
#include <OpenGLRenderer.h>
#include <OpenGLApp.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <SceneLoader.h>

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

int main(int argc, char** argv) {
    Config config{};
    config.title = "QuadWarp Streamer";
    config.openglMajorVersion = 4;
    config.openglMinorVersion = 3;
    config.enableVSync = false;
    config.showWindow = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'i', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 8);
    args::ValueFlag<int> renderStateIn(parser, "render", "Render state", {'r', "render-state"}, 0);
    args::ValueFlag<int> maxStepsIn(parser, "steps", "Max steps", {'m', "max-steps"}, 10);
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find("x");
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    int surfelSize = args::get(surfelSizeIn);
    int maxSteps = args::get(maxStepsIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(&screenWidth, &screenHeight);

    Scene scene = Scene();
    Camera camera = Camera(screenWidth, screenHeight);
    SceneLoader loader = SceneLoader();
    loader.loadScene(scenePath, scene, camera);

    guiManager->onRender([&](double now, double dt) {
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        int flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("", 0, flags);
        ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);
        glm::vec2 guiSize = winSize * glm::vec2(0.4f, 0.3f);
        ImGui::SetNextWindowSize(ImVec2(guiSize.x, guiSize.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 60), ImGuiCond_FirstUseEver);
        flags = 0;
        ImGui::Begin(config.title.c_str(), 0, flags);
        ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
        ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));
        ImGui::End();
    });

    // shaders
    Shader screenShader = Shader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
        // .fragmentCodePath = "../shaders/postprocessing/displayNormals.frag",
        // .fragmentCodePath = "../shaders/postprocessing/displayIDs.frag"
    });

    ComputeShader genQuadsShader({
        .computeCodePath = "./shaders/genQuads.comp"
    });

    int width = screenWidth / surfelSize;
    int height = screenHeight / surfelSize;

    GLuint vertexBuffer;
    int numVertices = width * height * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVertices * sizeof(Vertex), nullptr, GL_STATIC_DRAW);

    GLuint indexBuffer;
    int numTriangles = width * height * NUM_SUB_QUADS * 2;
    int indexBufferSize = numTriangles * 3;
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, indexBufferSize * sizeof(GLuint), nullptr, GL_STATIC_DRAW);

    genQuadsShader.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
    genQuadsShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
    genQuadsShader.unbind();

    RenderTarget renderTarget({
        .width = screenWidth,
        .height = screenHeight,
        .internalFormat = GL_RGBA16,
        .format = GL_RGBA,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    });

    // camera.setProjectionMatrix(glm::radians(120.0f), (float)screenWidth / (float)screenHeight, 0.1f, 1000.0f);

    // save camera view and projection matrices
    std::ofstream cameraFile;
    cameraFile.open("data/camera.bin", std::ios::out | std::ios::binary);
    glm::mat4 proj = camera.getProjectionMatrix();
    glm::mat4 view = camera.getViewMatrix();
    cameraFile.write(reinterpret_cast<const char*>(&proj), sizeof(glm::mat4));
    cameraFile.write(reinterpret_cast<const char*>(&view), sizeof(glm::mat4));
    cameraFile.close();

    glm::vec3 initialPosition = camera.position;

    unsigned int t = 0;
    float z = 0.0f;
    app.onRender([&](double now, double dt) {
        auto saveFrame = [&](glm::vec3 position, std::string label, unsigned int timestamp) {
            std::cout << "saving [" << label << "] t=" << std::to_string(timestamp) << std::endl;

            double startTime = glfwGetTime();

            camera.position = position;
            camera.updateViewMatrix();

            // render all objects in scene
            app.renderer->drawObjects(scene, camera);

            // render to screen
            // app.renderer->drawToScreen(screenShader);
            app.renderer->drawToRenderTarget(screenShader, renderTarget);

            genQuadsShader.bind();
            genQuadsShader.setMat4("viewInverse", glm::inverse(camera.getViewMatrix()));
            genQuadsShader.setMat4("projectionInverse", glm::inverse(camera.getProjectionMatrix()));
            genQuadsShader.setFloat("near", camera.near);
            genQuadsShader.setFloat("far", camera.far);
            genQuadsShader.setInt("surfelSize", surfelSize);
            app.renderer->gBuffer.positionBuffer.bind(0);
            app.renderer->gBuffer.normalsBuffer.bind(1);
            app.renderer->gBuffer.idBuffer.bind(2);
            app.renderer->gBuffer.depthBuffer.bind(3);
            genQuadsShader.dispatch(width, height, 1);
            genQuadsShader.unbind();

            std::cout << "\tRendering Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            // app.renderer->gBuffer.colorBuffer.saveTextureToPNG("imgs/color_" + label + "_" + std::to_string(timestamp) + ".png");
            // app.renderer->gBuffer.depthBuffer.saveDepthToFile("imgs/depth1.bin");
            renderTarget.bind();
            renderTarget.colorBuffer.saveTextureToPNG("imgs/color_" + label + "_" + std::to_string(timestamp) + ".png");
            renderTarget.unbind();

            std::cout << "\tSaving Texture Time: " << glfwGetTime() - startTime << "s" << std::endl;
            startTime = glfwGetTime();

            // std::ofstream depthFile;
            // depthFile.open("data/depth_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            std::ofstream verticesFile;
            verticesFile.open("data/vertices_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            // std::ofstream texCoordsFile;
            // texCoordsFile.open("data/tex_coords_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            std::ofstream indicesFile;
            indicesFile.open("data/indices_" + label + "_" + std::to_string(timestamp) + ".bin", std::ios::out | std::ios::binary);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
            GLvoid* pVertexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pVertexBuffer) {
                verticesFile.write(reinterpret_cast<const char*>(pVertexBuffer), numVertices * sizeof(Vertex));
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to save vertex buffer" << std::endl;
            }

            std::cout << "\tSaving Vertices Time: " << glfwGetTime() - startTime << "s" << std::endl;
            std::cout << "\t" << numVertices << " vertices" << std::endl;
            startTime = glfwGetTime();

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
            GLvoid* pIndexBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (pIndexBuffer) {
                indicesFile.write(reinterpret_cast<const char*>(pIndexBuffer), indexBufferSize * sizeof(unsigned int));
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                std::cerr << "Failed to save index buffer" << std::endl;
            }

            std::cout << "\tSaving Indices Time: " << glfwGetTime() - startTime << "s" << std::endl;
        };

        saveFrame(initialPosition + glm::vec3(0.0f, 0.0f, 0.0f + z), "center", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, 0.5f, 0.5f + z), "top_right_front", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, 0.5f, -0.5f + z), "top_right_back", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, 0.5f, 0.5f + z), "top_left_front", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, 0.5f, -0.5f + z), "top_left_back", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, -0.5f, 0.5f + z), "bottom_right_front", t);
        // saveFrame(initialPosition + glm::vec3(0.5f, -0.5f, -0.5f + z), "bottom_right_back", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, -0.5f, 0.5f + z), "bottom_left_front", t);
        // saveFrame(initialPosition + glm::vec3(-0.5f, -0.5f, -0.5f + z), "bottom_left_back", t);

        z -= 0.5f;

        t++;
        if (t >= maxSteps) {
            window->close();
        }
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
