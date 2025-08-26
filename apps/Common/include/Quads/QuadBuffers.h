#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <glm/glm.hpp>

#include <Buffer.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

namespace quasar {

struct QuadMapData {
    glm::vec3 normal;
    float depth;
    glm::ivec2 offset;
    uint32_t size;
    bool flattened;
};

struct QuadMapDataPacked {
    // Normal converted into spherical coordinates. 16 bits of padding + theta, phi (8 bits each) packed into 16 bits.
    uint32_t normalSpherical;
    // Full resolution depth. 32 bits used.
    float depth;
    // offset.x << 20 | offset.y << 8 (12 bits each) | size << 1 (6 bits) | flattened (1 bit). 31 bits used.
    uint32_t metadata;
}; // 96 bits total

class QuadBuffers {
public:
    size_t maxProxies;
    size_t numProxies;
    size_t maxDataSize;

    Buffer normalSphericalsBuffer;
    Buffer depthsBuffer;
    Buffer metadatasBuffer;

    QuadBuffers(size_t maxProxies);
    ~QuadBuffers() = default;

    void resize(size_t numProxies);

#ifdef GL_CORE
    size_t copyToCPU(std::vector<char>& outputData);
#endif
    size_t copyFromCPU(const std::vector<char>& inputData);

private:
#if defined(HAS_CUDA)
    CudaGLBuffer cudaBufferNormalSphericals;
    CudaGLBuffer cudaBufferDepths;
    CudaGLBuffer cudaBufferMetadatas;
#endif
};

} // namespace quasar

#endif // QUAD_BUFFERS_H
