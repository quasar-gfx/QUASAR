#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#if !defined(__ANDROID__)
#include <filesystem>
#endif

#include <spdlog/spdlog.h>

#include <Texture.h>
#include <Utils/FileIO.h>

#include <Compression/ZSTDCompressor.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

class DepthOffsets {
public:
    glm::uvec2 size;
    Texture buffer;

    DepthOffsets(const glm::uvec2 &size);
    ~DepthOffsets();

    unsigned int loadFromMemory(const char* data);
    unsigned int loadFromFile(const std::string &filename, unsigned int* numBytesLoaded = nullptr);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    unsigned int saveToMemory(std::vector<char> &compressedData);
    unsigned int saveToFile(const std::string &filename);
#endif

private:
    ZSTDCompressor compressor;

    std::vector<char> data;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResource;
#endif
};

#endif // DEPTH_OFFSETS_H
