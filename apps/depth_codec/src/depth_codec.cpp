#include <iostream>

#include <args.hxx>

#include <OpenGLApp.h>
#include <Renderers/ForwardRenderer.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>

#include <VideoTexture.h>
#include <DepthVideoTexture.h>
#include <PoseStreamer.h>

#define TEXTURE_PREVIEW_SIZE 250

#define VERTICES_IN_A_QUAD 4

const std::string DATA_PATH = "./";

enum class RenderState {
    MESH,
    POINTCLOUD
};

// Modify the convert32To16Bit function to accept a const float* and a size
std::vector<uint16_t> convert32To16Bit(const float* image, size_t size) {
    std::vector<uint16_t> convertedImage(size);
    for (size_t i = 0; i < size; ++i) {
        convertedImage[i] = static_cast<uint16_t>(image[i] * 65535.0f);
    }
    return convertedImage;
}

// std::vector<uint16_t> convert32To16Bit(const std::vector<float> &image) {
//     std::vector<uint16_t> convertedImage(image.size());
//     for (size_t i = 0; i < image.size(); ++i) {
//         float value = std::max(0.0f, std::min(1.0f, image[i]));  // Clamp value between 0 and 1
//         convertedImage[i] = static_cast<uint16_t>(value * 65535.0f);
//     }
//     return convertedImage;
// }

std::vector<uint8_t> bc4Compress(const std::vector<uint16_t> &image, size_t width, size_t height) {
    if (image.size() != width * height) {
        std::cerr << "Invalid image size for BC4 compression." << std::endl;
        return {};
    }
    const size_t BLOCK_SIZE = 8;
    std::vector<uint8_t> compressedData;
    
    for (size_t y = 0; y < height; y += BLOCK_SIZE) {
        for (size_t x = 0; x < width; x += BLOCK_SIZE) {
            std::vector<uint16_t> block(BLOCK_SIZE * BLOCK_SIZE);
            for (size_t by = 0; by < BLOCK_SIZE; ++by) {
                for (size_t bx = 0; bx < BLOCK_SIZE; ++bx) {
                    size_t ix = std::min(x + bx, width - 1);
                    size_t iy = std::min(y + by, height - 1);
                    block[by * BLOCK_SIZE + bx] = image[iy * width + ix];
                }
            }
            
            uint16_t minVal = *std::min_element(block.begin(), block.end());
            uint16_t maxVal = *std::max_element(block.begin(), block.end());
            
            compressedData.push_back(maxVal & 0xFF);
            compressedData.push_back((maxVal >> 8) & 0xFF);
            compressedData.push_back(minVal & 0xFF);
            compressedData.push_back((minVal >> 8) & 0xFF);
            
            std::vector<uint16_t> lookup(8);
            lookup[0] = maxVal;
            lookup[1] = minVal;
            for (int i = 1; i < 7; ++i) {
                lookup[i + 1] = ((7 - i) * maxVal + i * minVal) / 7;
            }
            
            std::vector<uint8_t> indices(BLOCK_SIZE * BLOCK_SIZE);
            for (size_t i = 0; i < BLOCK_SIZE * BLOCK_SIZE; ++i) {
                uint16_t pixel = block[i];
                size_t bestIndex = 0;
                uint16_t minDiff = std::numeric_limits<uint16_t>::max();
                for (size_t j = 0; j < 8; ++j) {
                    uint16_t diff = std::abs(static_cast<int>(pixel) - static_cast<int>(lookup[j]));
                    if (diff < minDiff) {
                        minDiff = diff;
                        bestIndex = j;
                    }
                }
                indices[i] = bestIndex;
            }

            for (size_t i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i += 8) {
                uint32_t packed = 0;
                for (size_t j = 0; j < 8; ++j) {
                    packed |= (indices[i + j] << (j * 3));
                }
                compressedData.push_back(packed & 0xFF);
                compressedData.push_back((packed >> 8) & 0xFF);
                compressedData.push_back((packed >> 16) & 0xFF);
            }
        }
    }
    
    return compressedData;
}

std::vector<uint16_t> bc4Decompress(const std::vector<uint8_t> &compressed, size_t width, size_t height) {
    const size_t BLOCK_SIZE = 8;
    std::vector<uint16_t> decompressed(width * height);
    size_t compressedIndex = 0;

    for (size_t y = 0; y < height; y += BLOCK_SIZE) {
        for (size_t x = 0; x < width; x += BLOCK_SIZE) {
            uint16_t maxVal = (compressed[compressedIndex + 1] << 8) | compressed[compressedIndex];
            uint16_t minVal = (compressed[compressedIndex + 3] << 8) | compressed[compressedIndex + 2];
            compressedIndex += 4;

            std::vector<uint16_t> lookup(8);
            lookup[0] = maxVal;
            lookup[1] = minVal;
            for (int i = 1; i < 7; ++i) {
                lookup[i + 1] = ((7 - i) * maxVal + i * minVal) / 7;
            }

            for (size_t i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i += 8) {
                uint32_t packed = compressed[compressedIndex] | (compressed[compressedIndex + 1] << 8) | (compressed[compressedIndex + 2] << 16);
                compressedIndex += 3;

                for (size_t j = 0; j < 8; ++j) {
                    uint8_t index = (packed >> (j * 3)) & 0x7;
                    size_t px = x + (i % BLOCK_SIZE) + j;
                    size_t py = y + (i / BLOCK_SIZE);
                    if (px < width && py < height) {
                        decompressed[py * width + px] = lookup[index];
                    }
                }
            }
        }
    }

    return decompressed;
}

double calculateMSE(const std::vector<uint16_t> &original, const std::vector<uint16_t> &decompressed, size_t width, size_t height, std::vector<uint16_t> &mseImage) {
    double totalMSE = 0.0;
    mseImage.resize(width * height);
    
    for (size_t i = 0; i < width * height; ++i) {
        // Normalize the values for MSE calculation
        double originalNorm = original[i] / 65535.0;
        double decompressedNorm = decompressed[i] / 65535.0;
        
        double diff = originalNorm - decompressedNorm;
        double squaredError = diff * diff;
        
        totalMSE += squaredError;
        
        // Calculate enhanced error for visualization
        double enhancedError = std::min(1.0, -std::log10(squaredError + 1e-10) / 10.0);
        mseImage[i] = static_cast<uint16_t>(enhancedError * 65535.0);
    }
    
    return totalMSE / (width * height);
}


int main(int argc, char** argv) {
    Config config{};
    config.title = "BC4 Compression";

    RenderState renderState = RenderState::POINTCLOUD;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Size of window", {'s', "size"}, "800x600");
    args::ValueFlag<std::string> scenePathIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::ValueFlag<bool> vsyncIn(parser, "vsync", "Enable VSync", {'v', "vsync"}, true);
    args::ValueFlag<int> surfelSizeIn(parser, "surfel", "Surfel size", {'z', "surfel-size"}, 1);
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
    size_t pos = sizeStr.find('x');
    config.width = std::stoi(sizeStr.substr(0, pos));
    config.height = std::stoi(sizeStr.substr(pos + 1));

    config.enableVSync = args::get(vsyncIn);

    std::string scenePath = args::get(scenePathIn);

    int surfelSize = args::get(surfelSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    unsigned int screenWidth, screenHeight;
    window->getSize(screenWidth, screenHeight);

    Scene scene = Scene();
    scene.backgroundColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    PerspectiveCamera camera = PerspectiveCamera(screenWidth, screenHeight);
    PerspectiveCamera origCamera = PerspectiveCamera(screenWidth, screenHeight);

    camera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    camera.updateViewMatrix();
    origCamera.setPosition(glm::vec3(0.0f, 3.0f, 10.0f));
    origCamera.updateViewMatrix();


    // shaders
    Shader screenShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayColor.frag"
    });

    Shader videoShader({
        .vertexCodePath = "../shaders/postprocessing/postprocess.vert",
        .fragmentCodePath = "../shaders/postprocessing/displayTexture.frag",
    });

    ComputeShader genPtCloudFromDepthShader({
        .computeCodePath = "./shaders/genPtCloudFromDepth.comp"
    });

    ComputeShader bc4DepthShader({
        .computeCodePath = "./shaders/bc4.comp"
    });

    ComputeShader copyShader({ // my buffer
        .computeCodePath = "./shaders/copy.comp"
    });

 
    unsigned int depthWidth = 2048, depthHeight = 2048;
    // std::vector<float> depthData(screenWidth * screenHeight);
    auto depthData = FileIO::loadBinaryFile(DATA_PATH + "depth.bin"); // 
    std::cout<< "depthData size: "<< depthData.size()<<std::endl;
    // std::memcpy(depthData.data(), depthFile.data(), depthFile.size());

    struct block{
        float max; // 32 - unit32
        float min;
        uint32_t arr[6]; // 32
    };

    bc4DepthShader.bind();
    bc4DepthShader.setVec2("depthMapSize", glm::vec2(depthWidth, depthHeight));
    bc4DepthShader.setVec2("bc4DepthSize", glm::vec2(depthWidth/8, depthHeight/8)); // 32+32+64*3 bits->bytes 2048/8 * 2048/)

    Buffer<block> bc4Buffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, (depthWidth/8 * depthHeight/8), nullptr); // 32+32+64*3 bits->bytes 2048/8 * 2048/8 = 

    if (depthData.empty()) {
        std::cerr << "Failed to load depth data from file." << std::endl;
        return 1;
    }

    if (depthData.size() != depthWidth * depthHeight * sizeof(float)) {
        std::cerr << "Unexpected depth data size. Expected " << (depthWidth * depthHeight * sizeof(float))
                << " bytes, but got " << depthData.size() << " bytes." << std::endl;
        return 1;
    }

    // Convert depth data to 16-bit
    // std::vector<float> floatDepthData(depthData.size() / sizeof(float));
    // std::memcpy(floatDepthData.data(), depthData.data(), depthData.size());
    // std::cout << "floartData size: " << floatDepthData.size() << std::endl;

    const float* floatDepthData = reinterpret_cast<const float*>(depthData.data());
    size_t floatDepthDataSize = depthData.size() / sizeof(float);


    std::vector<uint16_t> originalImage16 = convert32To16Bit(floatDepthData, floatDepthDataSize);
    std::cout << "[after 32 to 16]Original image size: " << originalImage16.size() << std::endl;

    // Compress the image using BC4
    std::vector<uint8_t> compressedImage = bc4Compress(originalImage16, depthWidth, depthHeight);
    std::cout << "Compressed image size: " << compressedImage.size() << std::endl;
    std::cout << "Expected compressed size: " << (depthWidth * depthHeight / 2) << std::endl;

    // Decompress the image
    std::vector<uint16_t> decompressedImage = bc4Decompress(compressedImage, depthWidth, depthHeight);

    // Calculate MSE and PSNR
    std::vector<uint16_t> mseImage;
    double totalMSE = calculateMSE(originalImage16, decompressedImage, depthWidth, depthHeight, mseImage);
    double psnr = 10.0 * log10(1.0 / totalMSE);

    std::cout << "Compression ratio: " << static_cast<double>(originalImage16.size() * sizeof(uint16_t)) / compressedImage.size() << ":1\n";
    std::cout << "Mean Squared Error (MSE) per pixel (normalized 0-1): " << totalMSE << "\n";
    std::cout << "Peak Signal-to-Noise Ratio (PSNR): " << psnr << " dB\n";
   
    if (originalImage16.empty()) {
        std::cerr << "Original image data is empty. Cannot create texture." << std::endl;
    }


    Texture depthTextureOriginal({
        .width = depthWidth,
        .height = depthHeight,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .data = reinterpret_cast<unsigned char*>(depthData.data()) // depthData
    });

    glGetError();
    std::cout << "Depth original Textures created successfully." << std::endl;
    

    // std::vector<float> depthDataDecompressed(depthWidth * depthHeight);
    // // fill with random values for now
    // for (size_t i = 0; i < depthDataDecompressed.size(); ++i) {
    //     depthDataDecompressed[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // }

    // Convert uint16_t back to float for texture creation, preserving the raw depth values
    std::vector<float> decompressedImageFloat(decompressedImage.size());
    for (size_t i = 0; i < decompressedImage.size(); ++i) {
        decompressedImageFloat[i] = static_cast<float>(decompressedImage[i]) / 65535.0f;
    }

    std::cout << "Decompressed image size: " << decompressedImage.size() << std::endl;
    std::cout << "Expected decompressed size: " << depthWidth * depthHeight << std::endl;
    std::cout << "decompressedImageFloat.data() address: " << static_cast<void*>(decompressedImageFloat.data()) << std::endl;


    if (decompressedImage.size() != depthWidth * depthHeight) {
        std::cerr << "Error: Decompressed image size mismatch." << std::endl;
        return 1;
    }

    bc4DepthShader.setTexture(depthTextureOriginal, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bc4Buffer);
    // run compute shader
    bc4DepthShader.dispatch(depthWidth /8 /16, depthHeight /8/ 16, 1); // each bloack, (now each pixel /16*16) - > bc4 comp
    bc4DepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);



    // Texture depthTextureDecompressed({
    //     .width = depthWidth,
    //     .height = depthHeight,
    //     .internalFormat = GL_R32F,
    //     .format = GL_RED,
    //     .type = GL_FLOAT,
    //     .wrapS = GL_CLAMP_TO_EDGE,
    //     .wrapT = GL_CLAMP_TO_EDGE,
    //     .minFilter = GL_NEAREST,
    //     .magFilter = GL_NEAREST,
    //     .data = reinterpret_cast<unsigned char*>(decompressedImageFloat.data())
    // });

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error after creating decompressed texture: " << error << std::endl;
        return 1;
    }

    std::cout << "Depth Decompressed Texture created successfully." << std::endl;



    int width = screenWidth / surfelSize;
    int height = screenHeight / surfelSize;

    int numVertices = width * height;

    int numTriangles = (width-1) * (height-1) * 2;
    int indexBufferSize = numTriangles * 3;

    
    Mesh mesh = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) }),
        .pointcloud = renderState == RenderState::POINTCLOUD,
        .pointSize = 7.5f,
        .usage = GL_DYNAMIC_DRAW
    });
    Node node = Node(&mesh);
    node.frustumCulled = false;
    scene.addChildNode(&node);

    Mesh meshDecompressed = Mesh({
        .vertices = std::vector<Vertex>(numVertices),
        .indices = std::vector<unsigned int>(indexBufferSize),
        .material = new UnlitMaterial({ .baseColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) }),
        .pointcloud = renderState == RenderState::POINTCLOUD,
        .pointSize = 5.0f,
        .usage = GL_DYNAMIC_DRAW
    });
    Node nodeDecompressed = Node(&meshDecompressed);
    nodeDecompressed.frustumCulled = false;
    scene.addChildNode(&nodeDecompressed);

    RenderStats renderStats;
    guiManager->onRender([&](double now, double dt) {
        static bool showFPS = true;
        static bool showUI = true;
        static bool showCaptureWindow = false;
        static bool showDepthPreview = true;

        glm::vec2 winSize = glm::vec2(screenWidth, screenHeight);

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
            ImGui::MenuItem("Depth Preview", 0, &showDepthPreview);
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
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Triangles Drawn: %d", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Total Draw Calls: %d", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Total Draw Calls: %d", renderStats.drawCalls);

            ImGui::Separator();

            glm::vec3 position = camera.getPosition();
            if (ImGui::InputFloat3("Camera Position", (float*)&position)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::InputFloat3("Camera Rotation", (float*)&rotation)) {
                camera.setRotationEuler(rotation);
            }

            ImGui::Separator();

            ImGui::RadioButton("Display Mesh", (int*)&renderState, 0);
            ImGui::RadioButton("Display Point Cloud", (int*)&renderState, 1);

            ImGui::Separator();

            ImGui::TextColored(ImVec4(0,0,0,1), "Original Depth Buffer");
            ImGui::TextColored(ImVec4(1,1,0,1), "Decompressed Depth Buffer");

            ImGui::Separator();

            ImGui::Checkbox("Show Original Depth", &node.visible);
            ImGui::Checkbox("Show Decompressed Depth", &nodeDecompressed.visible);

            ImGui::End();
        }

        flags = ImGuiWindowFlags_AlwaysAutoResize;

        if (showDepthPreview) {
            ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 40), ImGuiCond_FirstUseEver);
            ImGui::Begin("Original Depth", &showDepthPreview, flags);
            ImGui::Image((void*)(intptr_t)depthTextureOriginal.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();

            // ImGui::SetNextWindowPos(ImVec2(screenWidth - TEXTURE_PREVIEW_SIZE - 30, 70 + TEXTURE_PREVIEW_SIZE + 30), ImGuiCond_FirstUseEver);
            // ImGui::Begin("Decompressed Depth", &showDepthPreview, flags);
            // ImGui::Image((void*)(intptr_t)depthTextureDecompressed.ID, ImVec2(TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE), ImVec2(0, 1), ImVec2(1, 0));
            // ImGui::End();
        }
    });

    app.onResize([&](unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;

        renderer.resize(width, height);

        camera.aspect = (float)screenWidth / (float)screenHeight;
        camera.updateProjectionMatrix();
    });

    app.onRender([&](double now, double dt) {
        // handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = screenWidth / 2.0;
            static float lastY = screenHeight / 2.0;
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

        // handle keyboard input
        auto keys = window->getKeys();
        camera.processKeyboard(keys, dt);
        if (keys.ESC_PRESSED) {
            window->close();
        }

        // generate mesh for original and decompressed depth data
        genPtCloudFromDepthShader.bind();
        genPtCloudFromDepthShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
        genPtCloudFromDepthShader.setInt("surfelSize", surfelSize);
        
        {
            genPtCloudFromDepthShader.setMat4("view", origCamera.getViewMatrix());
            genPtCloudFromDepthShader.setMat4("projection", origCamera.getProjectionMatrix());
            genPtCloudFromDepthShader.setMat4("viewInverse", glm::inverse(origCamera.getViewMatrix()));
            genPtCloudFromDepthShader.setMat4("projectionInverse", glm::inverse(origCamera.getProjectionMatrix()));
        }
        {
            genPtCloudFromDepthShader.setTexture(depthTextureOriginal, 0);
        }
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
        }
        // dispatch compute shader to generate vertices and indices for mesh
        genPtCloudFromDepthShader.dispatch(width / 16, height / 16, 1);
        genPtCloudFromDepthShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // do it again with decompressed depth data:
        copyShader.bind();
        copyShader.setVec2("screenSize", glm::vec2(screenWidth, screenHeight));
        copyShader.setInt("surfelSize", surfelSize);
        {
            copyShader.setMat4("view", origCamera.getViewMatrix());
            copyShader.setMat4("projection", origCamera.getProjectionMatrix());
            copyShader.setMat4("viewInverse", glm::inverse(origCamera.getViewMatrix()));
            copyShader.setMat4("projectionInverse", glm::inverse(origCamera.getProjectionMatrix()));
        }
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meshDecompressed.vertexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshDecompressed.indexBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bc4Buffer);
        }
        // dispatch compute shader to generate vertices and indices for mesh
        copyShader.dispatch(width / 16, height / 16, 1);
        copyShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        // set render state
        mesh.pointcloud = renderState == RenderState::POINTCLOUD;
        meshDecompressed.pointcloud = renderState == RenderState::POINTCLOUD;

        // render all objects in scene
        renderStats = renderer.drawObjects(scene, camera);

        // render to screen
        screenShader.bind();
        screenShader.setBool("doToneMapping", false);
        renderer.drawToScreen(screenShader);
    });

    // run app loop (blocking)
    app.run();

    return 0;
}
