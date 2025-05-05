#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <Texture.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthOffsets {
public:
    struct Stats {
        double timeToCompressMs = 0.0f;
        double timeToDecompressMs = 0.0f;
    } stats;

    glm::uvec2 size;
    Texture buffer;

    DepthOffsets(const glm::uvec2 &size);
    ~DepthOffsets() = default;

    uint loadFromMemory(std::vector<char> &compressedData, bool decompress = true);
    uint loadFromFile(const std::string &filename, uint* numBytesLoaded = nullptr, bool compressed = true);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    uint saveToMemory(std::vector<char> &compressedData, bool compress = true);
    uint saveToFile(const std::string &filename);
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
