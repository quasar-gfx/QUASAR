#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#if !defined(__ANDROID__)
#include <filesystem>
#endif

#include <spdlog/spdlog.h>

#include <Texture.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthOffsets {
public:
    glm::uvec2 size;
    Texture buffer;

    DepthOffsets(const glm::uvec2 &size);
    ~DepthOffsets() = default;

    unsigned int loadFromMemory(std::vector<char> &compressedData, bool decompress = true);
    unsigned int loadFromFile(const std::string &filename, unsigned int* numBytesLoaded = nullptr, bool compressed = true);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    unsigned int saveToMemory(std::vector<char> &compressedData, bool compress = true);
    unsigned int saveToFile(const std::string &filename);
#endif

private:
    ZSTDCodec codec;

    std::vector<char> data;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    CudaGLImage cudaImage;
#endif
};

} // namespace quasar

#endif // DEPTH_OFFSETS_H
