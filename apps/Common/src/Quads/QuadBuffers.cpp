#include <spdlog/spdlog.h>

#include <Quads/QuadBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(size_t maxProxies)
    : maxProxies(maxProxies)
    , numProxies(maxProxies)
    , normalSphericalsBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , depthsBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(float),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , metadatasBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
#if defined(HAS_CUDA)
    , cudaBufferNormalSphericals(normalSphericalsBuffer)
    , cudaBufferDepths(depthsBuffer)
    , cudaBufferMetadatas(metadatasBuffer)
#endif
    , maxDataSize(sizeof(uint) + maxProxies * sizeof(QuadMapDataPacked))
{}

void QuadBuffers::resize(size_t numProxies) {
    this->numProxies = numProxies;
}

#ifdef GL_CORE
size_t QuadBuffers::copyToCPU(std::vector<char>& outputData) {
    outputData.resize(maxDataSize);
    size_t bufferOffset = 0;

    std::memcpy(outputData.data(), &numProxies, sizeof(uint));
    bufferOffset += sizeof(uint);

#if defined(HAS_CUDA)
    CudaGLBuffer::registerHostBuffer(outputData.data(), outputData.size());

    cudaBufferNormalSphericals.copyToHostAsync(outputData.data() + bufferOffset, numProxies * sizeof(uint));
    bufferOffset += numProxies * sizeof(uint);

    cudaBufferDepths.copyToHostAsync(outputData.data() + bufferOffset, numProxies * sizeof(float));
    bufferOffset += numProxies * sizeof(float);

    cudaBufferMetadatas.copyToHostAsync(outputData.data() + bufferOffset, numProxies * sizeof(uint));
    bufferOffset += numProxies * sizeof(uint);

    cudaBufferNormalSphericals.synchronize();
    CudaGLBuffer::unregisterHostBuffer(outputData.data());
#else
    void* ptr;

    normalSphericalsBuffer.bind();
    ptr = normalSphericalsBuffer.mapToCPU(GL_MAP_READ_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(outputData.data() + bufferOffset, ptr, numProxies * sizeof(uint));
        normalSphericalsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalSphericalsBuffer. Copying using setData");
        normalSphericalsBuffer.setData(numProxies, outputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint);

    depthsBuffer.bind();
    ptr = depthsBuffer.mapToCPU(GL_MAP_READ_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(outputData.data() + bufferOffset, ptr, numProxies * sizeof(float));
        depthsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map depthsBuffer. Copying using setData");
        depthsBuffer.setData(numProxies, outputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(float);

    metadatasBuffer.bind();
    ptr = metadatasBuffer.mapToCPU(GL_MAP_READ_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(outputData.data() + bufferOffset, ptr, numProxies * sizeof(uint));
        metadatasBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map metadatasBuffer. Copying using setData");
        metadatasBuffer.setData(numProxies, outputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint);
#endif

    // Resize output
    outputData.resize(bufferOffset);
    return bufferOffset;
}
#endif

size_t QuadBuffers::copyFromCPU(const std::vector<char>& inputData) {
    size_t bufferOffset = 0;
    void* ptr;

    numProxies = *reinterpret_cast<const uint*>(inputData.data());
    bufferOffset += sizeof(uint);

    normalSphericalsBuffer.bind();
    ptr = normalSphericalsBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, inputData.data() + bufferOffset, numProxies * sizeof(uint));
        normalSphericalsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalSphericalsBuffer. Copying using setData");
        normalSphericalsBuffer.setData(numProxies, inputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint);

    depthsBuffer.bind();
    ptr = depthsBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, inputData.data() + bufferOffset, numProxies * sizeof(float));
        depthsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map depthsBuffer. Copying using setData");
        depthsBuffer.setData(numProxies, inputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(float);

    metadatasBuffer.bind();
    ptr = metadatasBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, inputData.data() + bufferOffset, numProxies * sizeof(uint));
        metadatasBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map metadatasBuffer. Copying using setData");
        metadatasBuffer.setData(numProxies, inputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint);

    // Set new number of proxies
    resize(numProxies);
    return numProxies;
}
