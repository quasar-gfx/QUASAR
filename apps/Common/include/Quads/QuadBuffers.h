#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <glm/glm.hpp>

#include <Buffer.h>

#if defined(QUASAR_HAS_CUDA)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

namespace quasar {

struct QuadMapData {
    glm::vec3 normal;
    float depth;
    glm::uvec2 offset;
    uint32_t size;
    bool hasAlpha;
    bool flattened;
};

struct QuadMapDataPacked {
    // normal is converted into octahedral coordinates quantized to two 8 bit values.
    // depth is quantized to 16 bits.
    // (normal theta << 24) | (normal phi << 16) | (depth). 16 + 16 bits = 32 bits used.
    uint32_t normalAndDepth;
    // (offset.x << 20) | (offset.y << 8) | (size << 2) | (hasAlpha << 1) | (flattened). 12 + 12 + 5 + 1 + 1 bits = 32 bits used.
    uint32_t metadata;
}; // 64 bits total

class QuadBuffers {
public:
    uint32_t maxProxies = 0;
    uint32_t numProxies = 0;
    uint32_t numProxiesTransparent = 0;

    Buffer normalAndDepthBuffer;
    Buffer metadatasBuffer;

    QuadBuffers(uint32_t maxProxies);
    ~QuadBuffers() = default;

    void resize(uint32_t newNumProxies, uint32_t newnumProxiesTransparent);

#ifdef GL_CORE
    size_t writeToMemory(std::vector<char>& outputData, bool applyDeltaEncoding = true);
#endif
    size_t loadFromMemory(std::vector<char>& inputData, bool applyDeltaEncoding = true);

private:
#if defined(QUASAR_HAS_CUDA)
    CudaGLBuffer cudaBufferNormalAndDepth;
    CudaGLBuffer cudaBufferMetadatas;
#endif
};

} // namespace quasar

#endif // QUAD_BUFFERS_H
