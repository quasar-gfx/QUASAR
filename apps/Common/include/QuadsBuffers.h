#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <Buffer.h>
#include <Utils/FileIO.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

struct QuadMapDataPacked {
    // normal converted into spherical coordinates. theta, phi (16 bits each) packed into 32 bits
    unsigned int normalSpherical;
    // full resolution depth (32 bits)
    float depth;
    // x << 12 | y (12 bits each). 24 bits used
    unsigned int xy;
    // offset.x << 20 | offset.y << 8 (12 bits each) | size << 1 (5 bits) | flattened (1 bit). 30 bits used
    unsigned int offsetSizeFlattened;
}; // 128 bits total

class QuadBuffers {
public:
    unsigned int maxProxies;
    unsigned int numProxies;

    Buffer<unsigned int> normalSphericalsBuffer;
    Buffer<float> depthsBuffer;
    Buffer<unsigned int> xysBuffer;
    Buffer<unsigned int> offsetSizeFlattenedsBuffer;

    char* proxiesData;

    QuadBuffers(unsigned int maxProxies);
    ~QuadBuffers();

    void resize(unsigned int numProxies);

    unsigned int loadFromMemory(const char* data);
    unsigned int loadFromFile(const std::string &filename);
#ifdef GL_CORE
    unsigned int saveProxiesToFile(const std::string &filename);
    unsigned int updateProxiesDataBuffer();
#endif

private:
#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResourceNormalSphericals;
    cudaGraphicsResource* cudaResourceDepths;
    cudaGraphicsResource* cudaResourceXys;
    cudaGraphicsResource* cudaResourceOffsetSizeFlatteneds;
#endif
};

#endif // QUAD_BUFFERS_H
