#include <spdlog/spdlog.h>

#include <Quads/DepthOffsets.h>

using namespace quasar;

DepthOffsets::DepthOffsets(const glm::uvec2& size)
    : size(size)
    , buffer({
        .width = size.x,
        .height = size.y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
#if defined(HAS_CUDA)
    , cudaImage(buffer)
#endif
    , data(size.x * size.y * 4 * sizeof(uint16_t))
{}

#if defined(HAS_CUDA)
uint DepthOffsets::copyToMemory(std::vector<char>& outputData) {
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());
    outputData.resize(data.size());

    uint outputSize = data.size();
    memcpy(outputData.data(), data.data(), data.size());

    return outputSize;
}
#endif


uint DepthOffsets::loadFromMemory(std::vector<char>& inputData) {
    data = inputData;

#if defined(HAS_CUDA)
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());
#else
    buffer.setData(size.x, size.y, data.data());
#endif
    return data.size();
}
