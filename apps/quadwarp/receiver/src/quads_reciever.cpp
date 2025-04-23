#include <args/args.hxx>
#include <spdlog/spdlog.h>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <PostProcessing/ToneMapper.h>
#include <Recorder.h>
#include <CameraAnimator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <shaders_common.h>
#include <Networking/Socket.h>

#include <VideoTexture.h>
#include <Quads/QuadReceiver.h>

#define TEXTURE_PREVIEW_SIZE 500

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Quads Receiver";

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<std::string> quadServerAddressIn(parser, "quad-server", "Quad server address", {'q', "quad-address"}, "127.0.0.1");
    args::ValueFlag<int> quadServerPortIn(parser, "quad-port", "Quad server port", {'p', "quad-port"}, 9000);
    args::ValueFlag<std::string> videoURLIn(parser, "video", "Video URL", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> videoFormatIn(parser, "video-format", "Video format", {'g', "video-format"}, "mpegts");
    args::ValueFlag<std::string> colorFilePathIn(parser, "color-file", "Path to color texture file", {'f', "color-file"}, "../streamer/color.jpg");

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

    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    }

    // parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);

    std::string quadServerAddress = args::get(quadServerAddressIn);
    int quadServerPort = args::get(quadServerPortIn);
    std::string videoURL = args::get(videoURLIn);
    std::string videoFormat = args::get(videoFormatIn);
    std::string colorFilePath = args::get(colorFilePathIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize.x, windowSize.y);
    PerspectiveCamera remoteCamera(windowSize.x, windowSize.y);
    remoteCamera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    remoteCamera.updateViewMatrix();

    // Video texture for the received video frames
    VideoTexture videoTexture({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_BORDER,
        .wrapT = GL_CLAMP_TO_BORDER,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .hasBorder = true,
        .borderColor = glm::vec4(0.0f),
    }, videoURL, videoFormat);

    // post processing
    ToneMapper toneMapper;
    toneMapper.enableToneMapping(false);

    Recorder recorder(renderer, toneMapper, config.targetFramerate);

    // Setup for quad mesh generation
    const glm::uvec2 halfWindowSize = windowSize / 2u;
    MeshFromQuads meshFromQuads(windowSize);

    // Initialize color texture
    Texture colorTexture({
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .flipVertically = true,
        .path = colorFilePath
    });

    // Initialize quad receiver
    QuadReceiver quadReceiver(quadServerAddress, quadServerPort);

    // Initialize buffers for quad data and depth offsets
    unsigned int maxProxies = windowSize.x * windowSize.y * NUM_SUB_QUADS;
    QuadBuffers quadBuffers(maxProxies);
    
    const glm::uvec2 depthBufferSize = 2u * windowSize;
    DepthOffsets depthOffsets(depthBufferSize);
    
    // Network received data buffers
    std::vector<char> quadsData;
    std::vector<char> depthOffsetsData;
    
    // Create material with the color texture
    QuadMaterial* quadMaterial = new QuadMaterial({ 
        .baseColor = glm::vec4(1.0f), 
        .baseColorTexture = &videoTexture 
    });
    
    // Create mesh with the material
    Mesh* mesh = new Mesh({
        .maxVertices = maxProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
        .maxIndices = maxProxies * NUM_SUB_QUADS * INDICES_IN_A_QUAD,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = quadMaterial,
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    });
    
    // Create nodes for the mesh
    Node* node = new Node(mesh);
    node->frustumCulled = false;
    scene.addChildNode(node);

    // Create wireframe node with yellow color
    QuadMaterial* wireframeMaterial = new QuadMaterial({ 
        .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) 
    });

    // add a screen for the video. 

        Cube* videoScreen = new Cube({ 

        .material = new UnlitMaterial({ .baseColorTexture = &videoTexture }), 

        }); 

        Node* screen = new Node(videoScreen); 

        screen->setPosition(glm::vec3(0.0f, 0.0f, -2.0f)); 

        screen->setScale(glm::vec3(1.0f, 0.5f, 0.05f)); 

        screen->frustumCulled = false; 

        scene.addChildNode(screen); 
        
        scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

 
    
    Node* nodeWireframe = new Node(mesh);
    nodeWireframe->frustumCulled = false;
    nodeWireframe->wireframe = true;
    nodeWireframe->visible = false;
    nodeWireframe->overrideMaterial = wireframeMaterial;
    scene.addChildNode(nodeWireframe);
    
    // Statistics tracking
    unsigned int totalTriangles = 0;
    unsigned int totalProxies = 0;
    unsigned int totalDepthOffsets = 0;
    unsigned int numBytesProxies = 0;
    unsigned int numBytesDepthOffsets = 0;
    double loadDataTime = 0.0;
    double createMeshTime = 0.0;
    
    // UI state variables
    bool showWireframe = false;
    
    app.onResize([&](unsigned int width, unsigned int height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);
        camera.setAspect(windowSize.x, windowSize.y);
        camera.updateProjectionMatrix();
    });

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool saveAsHDR = false;
        static char fileNameBase[256] = "screenshot";
        static bool showVideoPreview = false;

        ImGui::NewFrame();

        unsigned int flags = 0;
        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &showFPS);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Frame Capture", 0, &showCaptureWindow);
            ImGui::MenuItem("Video Preview", 0, &showVideoPreview);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        if (showFPS) {
            ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
            flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
            ImGui::Begin("", 0, flags);
            ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
            ImGui::End();
        }

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %d", totalTriangles);
            else if (totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %d", totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %d", totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %d", renderStats.drawCalls);

            float proxySizeMB = static_cast<float>(numBytesProxies) / (1024.0f * 1024.0f);
            float depthOffsetSizeMB = static_cast<float>(numBytesDepthOffsets) / (1024.0f * 1024.0f);
            ImGui::TextColored(ImVec4(0,1,1,1), "Total Proxies: %d (%.3f MB)", totalProxies, proxySizeMB);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %d (%.3f MB)", totalDepthOffsets, depthOffsetSizeMB);

            ImGui::Separator();

            glm::vec3 position = glm::vec3(0.0f, 3.0f, 10.0f);//camera.getPosition();
            if (ImGui::DragFloat3("Camera Position", (float*)&position, 0.01f)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::DragFloat3("Camera Rotation", (float*)&rotation, 0.1f)) {
                camera.setRotationEuler(rotation);
            }
            ImGui::DragFloat("Movement Speed", &camera.movementSpeed, 0.05f, 0.1f, 20.0f);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to load data: %.3f ms", loadDataTime);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to create mesh: %.3f ms", createMeshTime);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Quad packets received: %d", quadReceiver.stats.quadPacketsReceived);
            ImGui::TextColored(ImVec4(0,0.5,0,1), "Avg network time: %.3f ms", 
                quadReceiver.stats.totalQuadNetworkTimeMs / std::max(1u, quadReceiver.stats.quadPacketsReceived));
            
            ImGui::Separator();

            ImGui::Text("Remote Pose ID: %lu", quadReceiver.getCurrentPoseID());
            ImGui::Text("Video URL: %s (%s)", videoURL.c_str(), videoFormat.c_str());
            ImGui::Text("Quad Server: %s:%d", quadServerAddress.c_str(), quadServerPort);

            ImGui::Separator();

            if (ImGui::Checkbox("Show Wireframe", &showWireframe)) {
                if (nodeWireframe) {
                    nodeWireframe->visible = showWireframe;
                }
            }

            ImGui::End();
        }

        if (showVideoPreview) {
            ImGui::SetNextWindowPos(ImVec2(windowSize.x - TEXTURE_PREVIEW_SIZE - 30, 40), ImGuiCond_FirstUseEver);
            flags = ImGuiWindowFlags_AlwaysAutoResize;
            ImGui::Begin("Raw Video Texture", &showVideoPreview, flags);
            ImGui::Image((void*)(intptr_t)videoTexture.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        if (showCaptureWindow) {
            ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin("Frame Capture", &showCaptureWindow);

            ImGui::Text("Base File Name:");
            ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
            std::string fileName = std::string(fileNameBase) + "." + std::to_string(static_cast<int>(window->getTime() * 1000.0f));

            ImGui::Checkbox("Save as HDR", &saveAsHDR);

            ImGui::Separator();

            if (ImGui::Button("Capture Current Frame")) {
                recorder.saveScreenshotToFile(fileName, saveAsHDR);
            }

            ImGui::End();
        }
        if (showCaptureWindow) {

        }
    });

    app.onRender([&](double now, double dt) {

        videoTexture.bind();
        spdlog::info("videotexture {}", videoTexture.draw());


        // Try to receive new quad data
        if (quadReceiver.receiveQuadData(quadsData, depthOffsetsData)) {
            double startTime = window->getTime();
            
            // Process received data
            numBytesProxies = quadsData.size();
            numBytesDepthOffsets = depthOffsetsData.size();
            
            // Load quad data into QuadBuffers using loadFromMemory
            // Using compressed parameter as false since we're receiving raw data over the network
            spdlog::info("Loading quad data ({} bytes) into QuadBuffers...", numBytesProxies);
            totalProxies = quadBuffers.loadFromMemory(quadsData, true);
            spdlog::info("Loaded {} proxies from quad data", totalProxies);
            
            // Load depth offsets data into DepthOffsets using loadFromMemory
            if (!depthOffsetsData.empty()) {
                spdlog::info("Loading depth offsets data ({} bytes)...", numBytesDepthOffsets);
                totalDepthOffsets = depthOffsets.loadFromMemory(depthOffsetsData, true);
                spdlog::info("Loaded {} depth offsets", totalDepthOffsets);
            }
            
            loadDataTime = timeutils::secondsToMillis(window->getTime() - startTime);
            
            // Update the mesh with the new data
            if (totalProxies > 0) {
                startTime = window->getTime();
                
                // Get texture dimensions - use colorTexture's dimensions
                const glm::vec2 gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);


                mesh->material = quadMaterial;
                
                // Clear previous data and update with new data
                spdlog::info("Appending {} proxies to mesh...", totalProxies);
                meshFromQuads.appendQuads(
                    gBufferSize,
                    totalProxies,
                    quadBuffers
                );
                
                spdlog::info("Creating mesh from proxies...");
                meshFromQuads.createMeshFromProxies(
                    gBufferSize,
                    totalProxies, 
                    depthOffsets,
                    remoteCamera,
                    *mesh
                );
                
                createMeshTime = timeutils::secondsToMillis(window->getTime() - startTime);
                
                // Update statistics
                auto meshBufferSizes = meshFromQuads.getBufferSizes();
                totalTriangles = meshBufferSizes.numIndices / 3;
                spdlog::info("Mesh created with {} triangles", totalTriangles);

            }
        }


        
        // Handle input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = windowSize.x / 2.0;
            static float lastY = windowSize.y / 2.0;
            
            if (!prevMouseLeftPressed && mouseButtons.LEFT_PRESSED) {
                dragging = true;
                prevMouseLeftPressed = true;
                auto cursorPos = window->getCursorPos();
                lastX = static_cast<float>(cursorPos.x);
                lastY = static_cast<float>(cursorPos.y);
            }
            
            if (prevMouseLeftPressed && !mouseButtons.LEFT_PRESSED) {
                dragging = false;
                prevMouseLeftPressed = false;
            }
            
            if (dragging) {
                auto cursorPos = window->getCursorPos();
                float xpos = static_cast<float>(cursorPos.x);
                float ypos = static_cast<float>(cursorPos.y);
                float xoffset = xpos - lastX;
                float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
                lastX = xpos;
                lastY = ypos;
                camera.processMouseMovement(xoffset, yoffset, true);
            }
        }
        
        auto keys = window->getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window->close();
        }
        auto scroll = window->getScrollOffset();
        camera.processScroll(scroll.y);

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        toneMapper.drawToScreen(renderer);
    });

    // Run app loop (blocking)
    app.run();

    // Clean up
    delete mesh;
    delete quadMaterial;
    delete wireframeMaterial;
    
    return 0;
}