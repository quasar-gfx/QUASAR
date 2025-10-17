#include <spdlog/spdlog.h>

#include <Quads/QuadBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(uint32_t maxProxies)
    : maxProxies(maxProxies)
    , normalAndDepthBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .maxElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , metadatasBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .maxElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
#if defined(QUASAR_HAS_CUDA)
    , cudaBufferNormalAndDepth(normalAndDepthBuffer)
    , cudaBufferMetadatas(metadatasBuffer)
#endif
{}

void QuadBuffers::resize(uint32_t newNumProxies, uint32_t newNumProxiesTransparent) {
    numProxies = newNumProxies;
    numProxiesTransparent = newNumProxiesTransparent;
    spdlog::debug("Resized QuadBuffers to {} proxies ({} transparent)", newNumProxies, newNumProxiesTransparent);
}

#ifdef GL_CORE
size_t QuadBuffers::writeToMemory(std::vector<char>& outputData, bool applyDeltaEncoding) {
    size_t outputSize = 2 * sizeof(uint32_t) + numProxies * sizeof(QuadMapDataPacked);
    if (outputData.size() != outputSize) {
        outputData.resize(outputSize);
    }

    size_t bufferOffset = 0;

    std::memcpy(outputData.data(), &numProxies, sizeof(uint32_t));
    bufferOffset += sizeof(uint32_t);

    std::memcpy(outputData.data() + bufferOffset, &numProxiesTransparent, sizeof(uint32_t));
    bufferOffset += sizeof(uint32_t);

    uint32_t* normalAndDepth = reinterpret_cast<uint32_t*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint32_t);

    uint32_t* metadatas = reinterpret_cast<uint32_t*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint32_t);

#if defined(QUASAR_HAS_CUDA)
    CudaGLBuffer::registerHostBuffer(outputData.data(), outputData.size());

    cudaBufferNormalAndDepth.copyToHostAsync(normalAndDepth, numProxies * sizeof(uint32_t));
    cudaBufferMetadatas.copyToHostAsync(metadatas, numProxies * sizeof(uint32_t));

    cudaBufferNormalAndDepth.synchronize();
    CudaGLBuffer::unregisterHostBuffer(outputData.data());
#else
    void* ptr;

    normalAndDepthBuffer.bind();
    ptr = normalAndDepthBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(normalAndDepth, ptr, numProxies * sizeof(uint32_t));
        normalAndDepthBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalAndDepthBuffer. Copying using getData");
        normalAndDepthBuffer.getData(outputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint32_t);

    metadatasBuffer.bind();
    ptr = metadatasBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(metadatas, ptr, numProxies * sizeof(uint32_t));
        metadatasBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map metadatasBuffer. Copying using getData");
        metadatasBuffer.getData(metadatas);
    }
    bufferOffset += numProxies * sizeof(uint32_t);
#endif

    if (applyDeltaEncoding) {
        // Apply delta encoding
        for (int i = numProxies - 1; i > 0; i--) {
            normalAndDepth[i] -= normalAndDepth[i - 1];
            metadatas[i] -= metadatas[i - 1];
        }
    }

    return bufferOffset;
}
#endif

size_t QuadBuffers::loadFromMemory(std::vector<char>& inputData, bool applyDeltaEncoding) {
    if (inputData.size() < 2 * sizeof(uint32_t)) {
        return 0;
    }

    size_t bufferOffset = 0;
    void* ptr;

    uint32_t newNumProxies = *reinterpret_cast<const uint32_t*>(inputData.data());
    bufferOffset += sizeof(uint32_t);

    uint32_t newNumProxiesTransparent = *reinterpret_cast<const uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += sizeof(uint32_t);

    uint32_t* normalAndDepth = reinterpret_cast<uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(uint32_t);

    uint32_t* metadatas = reinterpret_cast<uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(uint32_t);

    if (applyDeltaEncoding) {
        // Decode delta encoding
        for (int i = 1; i < newNumProxies; i++) {
            normalAndDepth[i] += normalAndDepth[i - 1];
            metadatas[i] += metadatas[i - 1];
        }
    }

    normalAndDepthBuffer.bind();
    ptr = normalAndDepthBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, normalAndDepth, newNumProxies * sizeof(uint32_t));
        normalAndDepthBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalAndDepthBuffer. Copying using setData");
        normalAndDepthBuffer.setData(newNumProxies, reinterpret_cast<char*>(normalAndDepth));
    }

    metadatasBuffer.bind();
    ptr = metadatasBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, metadatas, newNumProxies * sizeof(uint32_t));
        metadatasBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map metadatasBuffer. Copying using setData");
        metadatasBuffer.setData(newNumProxies, reinterpret_cast<char*>(metadatas));
    }

    // Set new number of proxies
    resize(newNumProxies, newNumProxiesTransparent);
    return bufferOffset;
}
