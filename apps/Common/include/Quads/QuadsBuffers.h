#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <glm/glm.hpp>

#include <Buffer.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

struct QuadMapData {
    glm::vec3 normal;
    float depth;
    glm::vec2 uv;
    glm::ivec2 offset;
    unsigned int size;
    bool flattened;
};

struct QuadMapDataPacked {
    // normal converted into spherical coordinates. theta, phi (16 bits each) packed into 32 bits
    unsigned int normalSpherical;
    // full resolution depth (32 bits)
    float depth;
    // offset.x << 20 | offset.y << 8 (12 bits each) | size << 1 (5 bits) | flattened (1 bit). 30 bits used
    unsigned int offsetSizeFlattened;
}; // 96 bits total

class QuadBuffers {
public:
    struct Stats {
        double timeToCompressMs = 0.0f;
        double timeToDecompressMs = 0.0f;
    } stats;

    unsigned int maxProxies;
    unsigned int numProxies;

    Buffer normalSphericalsBuffer;
    Buffer depthsBuffer;
    Buffer offsetSizeFlattenedsBuffer;

    QuadBuffers(unsigned int maxProxies);
    ~QuadBuffers() = default;

    void resize(unsigned int numProxies);

    unsigned int loadFromMemory(const std::vector<char> &compressedData, bool decompress = true);
    unsigned int loadFromFile(const std::string &filename, unsigned int* numBytesLoaded = nullptr, bool compressed = true);
#ifdef GL_CORE
    unsigned int saveToMemory(std::vector<char> &compressedData, bool compress = true);
    unsigned int saveToFile(const std::string &filename);
    unsigned int updateDataBuffer();
#endif

private:
    ZSTDCodec codec;

    std::vector<char> data;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    CudaGLBuffer cudaBufferNormalSphericals;
    CudaGLBuffer cudaBufferDepths;
    CudaGLBuffer cudaBufferOffsetSizeFlatteneds;
#endif
};

#endif // QUAD_BUFFERS_H
