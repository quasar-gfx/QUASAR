#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <lz4_stream/lz4_stream.h>

#include <glm/glm.hpp>

#include <Buffer.h>
#include <Utils/FileIO.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
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
        double timeToCompressionMs = 0.0f;
        double timeToDecompressionMs = 0.0f;
    } stats;

    unsigned int maxProxies;
    unsigned int numProxies;

    Buffer<unsigned int> normalSphericalsBuffer;
    Buffer<float> depthsBuffer;
    Buffer<unsigned int> offsetSizeFlattenedsBuffer;

    std::vector<uint8_t> data;

    QuadBuffers(unsigned int maxProxies);
    ~QuadBuffers();

    void resize(unsigned int numProxies);

    unsigned int loadFromMemory(const char* data);
    unsigned int loadFromFile(const std::string &filename, unsigned int* numBytesLoaded = nullptr);
#ifdef GL_CORE
    unsigned int saveToMemory(std::vector<char> &compressedData, bool doLZ4 = true);
    unsigned int saveToFile(const std::string &filename);
    unsigned int updateDataBuffer();
#endif

private:
    LZ4F_dctx* dctx = nullptr;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResourceNormalSphericals;
    cudaGraphicsResource* cudaResourceDepths;
    cudaGraphicsResource* cudaResourceOffsetSizeFlatteneds;
#endif
};

#endif // QUAD_BUFFERS_H
