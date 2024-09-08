#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

std::vector<float> readDepthBuffer(const std::string& filename, int width, int height) {
    std::vector<float> depthBuffer(width * height);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return depthBuffer;
    }
    file.read(reinterpret_cast<char*>(depthBuffer.data()), width * height * sizeof(float));
    file.close();
    return depthBuffer;
}

void normalizeDepthBuffer(std::vector<float>& depthBuffer, int width, int height) {
    float minDepth = INFINITY;
    float maxDepth = -INFINITY;
    for (float depth : depthBuffer) {
        if (depth > 0.0f) {
            if (depth < minDepth) minDepth = depth;
            if (depth > maxDepth) maxDepth = depth;
        }
    }
    for (float& depth : depthBuffer) {
        if (depth > 0.0f) {
            depth = (depth - minDepth) / (maxDepth - minDepth);
        } else {
            depth = 0.0f;
        }
    }
}

void renderDepthBuffer(const std::vector<float>& depthBuffer, int width, int height) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return;
    }
    GLFWwindow* window = glfwCreateWindow(width, height, "Depth Buffer Visualization", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create window!" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW!" << std::endl;
        glfwTerminate();
        return;
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_POINTS);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float depth = depthBuffer[y * width + x];
                glColor3f(depth, depth, depth);
                glVertex2i(x, height - y);
            }
        }
        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

std::vector<uint16_t> convert32To16Bit(const std::vector<float> &image) {
    std::vector<uint16_t> convertedImage(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        convertedImage[i] = static_cast<uint16_t>(image[i] * 65535.0f);
    }
    return convertedImage;
}

std::vector<uint8_t> bc4Compress(const std::vector<uint16_t> &image, size_t width, size_t height) {
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
        // Normalize pixel values to 0-1 range
        double originalNorm = original[i] / 65535.0;
        double decompressedNorm = decompressed[i] / 65535.0;
        
        // Calculate squared error using normalized values
        double diff = originalNorm - decompressedNorm;
        double squaredError = diff * diff;
        
        // Accumulate total MSE
        totalMSE += squaredError;
        
        // Enhanced visualization: use log scale and amplify
        // Note: This is only for visualization, not for the actual MSE calculation
        double enhancedError = std::min(1.0, -std::log10(squaredError + 1e-10) / 10.0);
        mseImage[i] = static_cast<uint16_t>(enhancedError * 65535.0);
    }
    
    // Return the average MSE (which is normalized, as we used normalized pixel values)
    return totalMSE / (width * height);
}

// double calculateMSE(const std::vector<uint16_t> &original, const std::vector<uint16_t> &decompressed, size_t width, size_t height, std::vector<uint16_t> &mseImage) {
//     double totalMSE = 0.0;
//     mseImage.resize(width * height);
    
//     for (size_t i = 0; i < width * height; ++i) {
//         int diff = static_cast<int>(original[i]) - static_cast<int>(decompressed[i]);
//         double squaredError = diff * diff;
//         totalMSE += squaredError;
//         mseImage[i] = static_cast<uint16_t>(std::min(65535.0, std::sqrt(squaredError) * 1024));
//     }
    
//     return totalMSE / (width * height);
// }

void displayImagesWithOpenGL(const std::vector<uint16_t> &original, const std::vector<uint16_t> &decompressed, const std::vector<uint16_t> &mse, size_t width, size_t height) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }

    GLFWwindow *window = glfwCreateWindow(1200, 400, "BC4 Compression - Image Comparison", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        return;
    }
    
    auto createTexture = [](const std::vector<uint16_t> &image, size_t width, size_t height) {
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        
        // Convert uint16_t to float for normalization
        std::vector<float> normalizedImage(image.size());
        for (size_t i = 0; i < image.size(); ++i) {
            normalizedImage[i] = image[i] / 65535.0f;
        }
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, normalizedImage.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        return textureID;
    };

    GLuint originalTex = createTexture(original, width, height);
    GLuint decompressedTex = createTexture(decompressed, width, height);
    GLuint mseTex = createTexture(mse, width, height);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        
        auto drawTexture = [](GLuint tex, float x1, float x2) {
            glBindTexture(GL_TEXTURE_2D, tex);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(x1, -0.9f);
            glTexCoord2f(1, 0); glVertex2f(x2, -0.9f);
            glTexCoord2f(1, 1); glVertex2f(x2, 0.9f);
            glTexCoord2f(0, 1); glVertex2f(x1, 0.9f);
            glEnd();
        };

        drawTexture(originalTex, -1.0f, -0.33f);
        drawTexture(decompressedTex, -0.33f, 0.33f);
        drawTexture(mseTex, 0.33f, 1.0f);
        
        // Add labels
        glColor3f(1.0f, 1.0f, 1.0f);
        glRasterPos2f(-0.9f, -0.95f);
        glfwSetWindowTitle(window, "BC4 Compression - Original | Decompressed | MSE");
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteTextures(1, &originalTex);
    glDeleteTextures(1, &decompressedTex);
    glDeleteTextures(1, &mseTex);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    std::string inputFile = "depth.bin";
    std::string compressedFile = "depth_compressed.bin";
    
    size_t width = 2048;
    size_t height = 2048;
    
    // Read and normalize the original depth buffer
    std::vector<float> originalImage = readDepthBuffer(inputFile, width, height);
    normalizeDepthBuffer(originalImage, width, height);
    
    // Convert to 16-bit representation
    std::vector<uint16_t> originalImage16 = convert32To16Bit(originalImage);
    
    // Compress the image using BC4
    std::vector<uint8_t> compressedImage = bc4Compress(originalImage16, width, height);
    
    // Save the compressed data
    std::ofstream compressedOut(compressedFile, std::ios::binary);
    compressedOut.write(reinterpret_cast<char*>(compressedImage.data()), compressedImage.size());
    compressedOut.close();
    
    // Print compression statistics
    std::cout << "Original uncompressed size: " << originalImage16.size() * sizeof(uint16_t) << " bytes\n";
    std::cout << "Compressed size: " << compressedImage.size() << " bytes\n";
    std::cout << "Compressed data saved to: " << compressedFile << "\n";
    
    double compressionRatio = static_cast<double>(originalImage16.size() * sizeof(uint16_t)) / compressedImage.size();
    std::cout << "Compression ratio: " << compressionRatio << ":1\n";
    
    // Decompress the image
    std::vector<uint16_t> decompressedImage = bc4Decompress(compressedImage, width, height);
    
    // Calculate normalized MSE
    std::vector<uint16_t> mseImage;
    double totalMSE = calculateMSE(originalImage16, decompressedImage, width, height, mseImage);
    std::cout << "Mean Squared Error (MSE) per pixel (normalized 0-1): " << totalMSE << "\n";

    // Calculate and display PSNR
    double psnr = 10.0 * log10(1.0 / totalMSE);  // Using 1.0 as max value since we're normalized
    std::cout << "Peak Signal-to-Noise Ratio (PSNR): " << psnr << " dB\n";
    
    std::cout << "\nDisplaying images:\n";
    std::cout << "Image 1: Original (uncompressed)\n";
    std::cout << "Image 2: Decompressed\n";
    std::cout << "Image 3: MSE of original and decompressed\n";
    
    // Display the images using OpenGL
    displayImagesWithOpenGL(originalImage16, decompressedImage, mseImage, width, height);
    
    return 0;
}