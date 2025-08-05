#include <spdlog/spdlog.h>

#include <Quads/QuadBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(uint maxProxies)
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
    , cudaBuffermetadatas(metadatasBuffer)
#endif
    , data(sizeof(uint) + maxProxies * sizeof(QuadMapDataPacked))
{}

void QuadBuffers::resize(uint numProxies) {
    this->numProxies = numProxies;
}

#ifdef GL_CORE
uint QuadBuffers::copyToMemory(std::vector<char>& outputData) {
    uint dataSize = updateDataBuffer();
    outputData.resize(dataSize);

    uint outputSize = dataSize;
    memcpy(outputData.data(), data.data(), dataSize);

    return outputSize;
}

uint QuadBuffers::updateDataBuffer() {
    uint bufferOffset = 0;

#if defined(HAS_CUDA)
    void* cudaPtr;

    memcpy(data.data(), &numProxies, sizeof(uint));
    bufferOffset += sizeof(uint);

    cudaPtr = cudaBufferNormalSphericals.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(uint), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(uint);

    cudaPtr = cudaBufferDepths.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(float), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(float);

    cudaPtr = cudaBuffermetadatas.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(uint), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(uint);
#else
    memcpy(data.data(), &numProxies, sizeof(uint));
    bufferOffset += sizeof(uint);

    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint);

    depthsBuffer.bind();
    depthsBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(float);

    metadatasBuffer.bind();
    metadatasBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint);
#endif
    return bufferOffset;
}
#endif

uint QuadBuffers::loadFromMemory(const std::vector<char>& inputData) {
    data = inputData;

    uint bufferOffset = 0;

    numProxies = *reinterpret_cast<const uint*>(data.data());
    bufferOffset += sizeof(uint);

    auto normalSphericalsPtr = reinterpret_cast<const uint*>(data.data() + bufferOffset);
    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.setData(numProxies, normalSphericalsPtr);
    bufferOffset += numProxies * sizeof(uint);

    auto depthsPtr = reinterpret_cast<const float*>(data.data() + bufferOffset);
    depthsBuffer.bind();
    depthsBuffer.setData(numProxies, depthsPtr);
    bufferOffset += numProxies * sizeof(float);

    auto metadatasPtr = reinterpret_cast<const uint*>(data.data() + bufferOffset);
    metadatasBuffer.bind();
    metadatasBuffer.setData(numProxies, metadatasPtr);

    return numProxies;
}

