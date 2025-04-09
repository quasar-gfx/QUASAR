#include <Quads/DepthOffsets.h>

using namespace quasar;

DepthOffsets::DepthOffsets(const glm::uvec2 &size)
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
        .magFilter = GL_NEAREST
    })
#if !defined(__APPLE__) && !defined(__ANDROID__)
    , cudaImage(buffer)
#endif
    , data(size.x * size.y * 4 * sizeof(uint16_t)) {

}

unsigned int DepthOffsets::loadFromMemory(std::vector<char> &compressedData, bool decompress) {
    if (decompress) {
        codec.decompress(compressedData, data);
    }
    else {
        data = compressedData;
    }

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());
#else
    buffer.setData(size.x, size.y, data.data());
#endif
    return data.size();
}

unsigned int DepthOffsets::loadFromFile(const std::string &filename, unsigned int* numBytesLoaded, bool compressed) {
#if !defined(__ANDROID__)
    if (!std::filesystem::exists(filename)) {
        spdlog::error("File {} does not exist", filename);
        return 0;
    }
#endif

    auto depthOffsetsCompressed = FileIO::loadBinaryFile(filename, numBytesLoaded);
    return loadFromMemory(depthOffsetsCompressed, compressed);
}

#if !defined(__APPLE__) && !defined(__ANDROID__)
unsigned int DepthOffsets::saveToMemory(std::vector<char> &compressedData, bool compress) {
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());
    compressedData.resize(data.size());
    if (!compress) {
        return codec.compress(data.data(), compressedData, data.size());
    }
    memcpy(compressedData.data(), data.data(), data.size());
    return data.size();
}

unsigned int DepthOffsets::saveToFile(const std::string &filename) {
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());

    std::vector<char> compressedData;
    unsigned int outputSize = saveToMemory(compressedData);

    std::ofstream quadsFile(filename + ".zstd", std::ios::binary);
    quadsFile.write(compressedData.data(), outputSize);
    quadsFile.close();

    return outputSize;
}
#endif
