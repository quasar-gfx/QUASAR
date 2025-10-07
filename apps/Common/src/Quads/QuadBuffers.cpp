#include <spdlog/spdlog.h>

#include <Quads/QuadBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(uint32_t maxProxies)
    : maxProxies(maxProxies)
    , numProxies(maxProxies)
    , normalSphericalsBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .maxElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , depthsBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(float),
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
    , cudaBufferNormalSphericals(normalSphericalsBuffer)
    , cudaBufferDepths(depthsBuffer)
    , cudaBufferMetadatas(metadatasBuffer)
#endif
    , maxDataSize(sizeof(uint32_t) + maxProxies * sizeof(QuadMapDataPacked))
{}

void QuadBuffers::resize(uint32_t newNumProxies) {
    numProxies = newNumProxies;
    spdlog::debug("Resized QuadBuffers to {} proxies", newNumProxies);
}

#ifdef GL_CORE
size_t QuadBuffers::writeToMemory(std::vector<char>& outputData, bool applyDeltaEncoding) {
    size_t outputSize = sizeof(uint32_t) + numProxies * sizeof(QuadMapDataPacked);
    if (outputData.size() != outputSize) {
        outputData.resize(outputSize);
    }

    size_t bufferOffset = 0;

    std::memcpy(outputData.data(), &numProxies, sizeof(uint32_t));
    bufferOffset += sizeof(uint32_t);

    uint32_t* normalSphericals = reinterpret_cast<uint32_t*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint32_t);

    float* depths = reinterpret_cast<float*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(float);

    uint32_t* metadatas = reinterpret_cast<uint32_t*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint32_t);

#if defined(QUASAR_HAS_CUDA)
    CudaGLBuffer::registerHostBuffer(outputData.data(), outputData.size());

    cudaBufferNormalSphericals.copyToHostAsync(normalSphericals, numProxies * sizeof(uint32_t));
    cudaBufferDepths.copyToHostAsync(depths, numProxies * sizeof(float));
    cudaBufferMetadatas.copyToHostAsync(metadatas, numProxies * sizeof(uint32_t));

    cudaBufferNormalSphericals.synchronize();
    CudaGLBuffer::unregisterHostBuffer(outputData.data());
#else
    void* ptr;

    normalSphericalsBuffer.bind();
    ptr = normalSphericalsBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(normalSphericals, ptr, numProxies * sizeof(uint32_t));
        normalSphericalsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalSphericalsBuffer. Copying using getData");
        normalSphericalsBuffer.getData(normalSphericals);
    }
    bufferOffset += numProxies * sizeof(uint32_t);

    depthsBuffer.bind();
    ptr = depthsBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(depths, ptr, numProxies * sizeof(float));
        depthsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map depthsBuffer. Copying using getData");
        depthsBuffer.getData(depths);
    }
    bufferOffset += numProxies * sizeof(float);

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
            normalSphericals[i] -= normalSphericals[i - 1];
            depths[i] -= depths[i - 1];
            metadatas[i] -= metadatas[i - 1];
        }
    }

    return bufferOffset;
}
#endif

size_t QuadBuffers::loadFromMemory(std::vector<char>& inputData, bool applyDeltaEncoding) {
    if (inputData.size() < sizeof(uint32_t)) {
        return 0;
    }

    size_t bufferOffset = 0;
    void* ptr;

    uint32_t newNumProxies = *reinterpret_cast<const uint32_t*>(inputData.data());
    bufferOffset += sizeof(uint32_t);

    uint32_t* normalSphericals = reinterpret_cast<uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(uint32_t);

    float* depths = reinterpret_cast<float*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(float);

    uint32_t* metadatas = reinterpret_cast<uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(uint32_t);

    if (applyDeltaEncoding) {
        // Decode delta encoding
        for (int i = 1; i < newNumProxies; i++) {
            normalSphericals[i] += normalSphericals[i - 1];
            depths[i] += depths[i - 1];
            metadatas[i] += metadatas[i - 1];
        }
    }

    normalSphericalsBuffer.bind();
    ptr = normalSphericalsBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, normalSphericals, newNumProxies * sizeof(uint32_t));
        normalSphericalsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalSphericalsBuffer. Copying using setData");
        normalSphericalsBuffer.setData(newNumProxies, reinterpret_cast<char*>(normalSphericals));
    }

    depthsBuffer.bind();
    ptr = depthsBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, depths, newNumProxies * sizeof(float));
        depthsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map depthsBuffer. Copying using setData");
        depthsBuffer.setData(newNumProxies, reinterpret_cast<char*>(depths));
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
    resize(newNumProxies);
    return bufferOffset;
}
