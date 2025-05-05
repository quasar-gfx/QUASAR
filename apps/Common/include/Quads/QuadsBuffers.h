#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <glm/glm.hpp>

#include <Buffer.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

namespace quasar {

struct QuadMapData {
    glm::vec3 normal;
    float depth;
    glm::ivec2 offset;
    uint size;
    bool flattened;
};

struct QuadMapDataPacked {
    // normal converted into spherical coordinates. 16 bits of padding + theta, phi (8 bits each) packed into 16 bits.
    uint normalSpherical;
    // full resolution depth. 32 bits used.
    float depth;
    // offset.x << 20 | offset.y << 8 (12 bits each) | size << 1 (6 bits) | flattened (1 bit). 31 bits used.
    uint offsetSizeFlattened;
}; // 96 bits total

class QuadBuffers {
public:
    struct Stats {
        double timeToCompressMs = 0.0f;
        double timeToDecompressMs = 0.0f;
    } stats;

    uint maxProxies;
    uint numProxies;

    Buffer normalSphericalsBuffer;
    Buffer depthsBuffer;
    Buffer offsetSizeFlattenedsBuffer;

    QuadBuffers(uint maxProxies);
    ~QuadBuffers() = default;

    void resize(uint numProxies);

    uint loadFromMemory(const std::vector<char> &compressedData, bool decompress = true);
    uint loadFromFile(const std::string &filename, uint* numBytesLoaded = nullptr, bool compressed = true);
#ifdef GL_CORE
    uint saveToMemory(std::vector<char> &compressedData, bool compress = true);
    uint saveToFile(const std::string &filename);
    uint updateDataBuffer();
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

} // namespace quasar

#endif // QUAD_BUFFERS_H
