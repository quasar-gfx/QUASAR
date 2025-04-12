#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <Texture.h>
#include <Utils/FileIO.h>
#include <Codec/LZ4Codec.h>

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

    unsigned int loadFromMemory(std::vector<char> &compressedData, bool decompress = true);
    unsigned int loadFromFile(const std::string &filename, unsigned int* numBytesLoaded = nullptr, bool compressed = true);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    unsigned int saveToMemory(std::vector<char> &compressedData, bool compress = true);
    unsigned int saveToFile(const std::string &filename);
#endif

private:
    LZ4Codec codec;

    std::vector<char> data;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    CudaGLImage cudaImage;
#endif
};

} // namespace quasar

#endif // DEPTH_OFFSETS_H
