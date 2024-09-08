#ifndef BC4_DEPTH_H
#define BC4_DEPTH_H
#include <vector>
#include <string>
#include <cstdint>
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Function declarations
std::vector<float> readDepthBuffer(const std::string& filename, int width, int height);
void normalizeDepthBuffer(std::vector<float>& depthBuffer);
std::vector<uint16_t> convert32To16Bit(const std::vector<float> &image);
std::vector<uint8_t> bc4Compress(const std::vector<uint16_t> &image, size_t width, size_t height);
std::vector<uint16_t> bc4Decompress(const std::vector<uint8_t> &compressed, size_t width, size_t height);
double calculateMSE(const std::vector<uint16_t> &original, const std::vector<uint16_t> &decompressed, size_t width, size_t height, std::vector<uint16_t> &mseImage);
void displayImagesWithOpenGL(const std::vector<uint16_t> &original, const std::vector<uint16_t> &decompressed, const std::vector<uint16_t> &mse, size_t width, size_t height);

#endif // BC4_DEPTH_H