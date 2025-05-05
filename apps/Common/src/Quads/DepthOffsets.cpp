#if !defined(__ANDROID__)
#include <filesystem>
#endif

#include <spdlog/spdlog.h>

#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

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

uint DepthOffsets::loadFromMemory(std::vector<char> &compressedData, bool decompress) {
    double startTime = timeutils::getTimeMicros();
    if (decompress) {
        codec.decompress(compressedData, data);
    }
    else {
        data = compressedData;
    }
    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());
#else
    buffer.setData(size.x, size.y, data.data());
#endif
    return data.size();
}

uint DepthOffsets::loadFromFile(const std::string &filename, uint* numBytesLoaded, bool compressed) {
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
uint DepthOffsets::saveToMemory(std::vector<char> &compressedData, bool compress) {
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());
    compressedData.resize(data.size());

    double startTime = timeutils::getTimeMicros();
    uint outputSize = data.size();
    if (compress) {
        outputSize = codec.compress(data.data(), compressedData, data.size());
        compressedData.resize(outputSize);
    }
    else {
        memcpy(compressedData.data(), data.data(), data.size());
    }
    stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return outputSize;
}

uint DepthOffsets::saveToFile(const std::string &filename) {
    cudaImage.copyToArray(size.x * 4 * sizeof(uint16_t), size.y, size.x * 4 * sizeof(uint16_t), data.data());

    std::vector<char> compressedData;
    uint outputSize = saveToMemory(compressedData);

    std::ofstream depthOffsetsFile(filename + ".zstd", std::ios::binary);
    depthOffsetsFile.write(compressedData.data(), outputSize);
    depthOffsetsFile.close();

    return outputSize;
}
#endif
