#include <QuadsBuffers.h>

QuadBuffers::QuadBuffers(unsigned int maxProxies)
        : maxProxies(maxProxies)
        , numProxies(maxProxies)
        , normalSphericalsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr)
        , depthsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr)
        , xysBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr)
        , offsetSizeFlattenedsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr) {
    proxiesData = new char[sizeof(unsigned int) + maxProxies * sizeof(QuadMapDataPacked)];

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudautils::checkCudaDevice();
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceNormalSphericals, normalSphericalsBuffer, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceDepths, depthsBuffer, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceXys, xysBuffer, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceOffsetSizeFlatteneds, offsetSizeFlattenedsBuffer, cudaGraphicsRegisterFlagsNone));
#endif
}

QuadBuffers::~QuadBuffers() {
    delete[] proxiesData;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceXys));
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceOffsetSizeFlatteneds));
#endif
}

void QuadBuffers::resize(unsigned int numProxies) {
    this->numProxies = numProxies;
}

unsigned int QuadBuffers::loadFromMemory(const char* data) {
    unsigned int bufferOffset = 0;

    numProxies = *reinterpret_cast<const unsigned int*>(data);
    bufferOffset += sizeof(unsigned int);

    auto normalSphericalsPtr = reinterpret_cast<const unsigned int*>(data + bufferOffset);
    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.setData(numProxies, normalSphericalsPtr);
    bufferOffset += numProxies * sizeof(unsigned int);

    auto depthsPtr = reinterpret_cast<const float*>(data + bufferOffset);
    depthsBuffer.bind();
    depthsBuffer.setData(numProxies, depthsPtr);
    bufferOffset += numProxies * sizeof(float);

    auto xysPtr = reinterpret_cast<const unsigned int*>(data + bufferOffset);
    xysBuffer.bind();
    xysBuffer.setData(numProxies, xysPtr);
    bufferOffset += numProxies * sizeof(unsigned int);

    auto offsetSizeFlattenedsPtr = reinterpret_cast<const unsigned int*>(data + bufferOffset);
    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.setData(numProxies, offsetSizeFlattenedsPtr);

    return numProxies;
}

unsigned int QuadBuffers::loadFromFile(const std::string &filename) {
    auto quadProxiesData = FileIO::loadBinaryFile(filename);
    return loadFromMemory(quadProxiesData.data());
}

#ifdef GL_CORE
unsigned int QuadBuffers::updateProxiesDataBuffer() {
    unsigned int bufferOffset = 0;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    void* cudaPtr;
    size_t size;

    memcpy(proxiesData, &numProxies, sizeof(unsigned int));
    bufferOffset += sizeof(unsigned int);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaMemcpy(proxiesData + bufferOffset, cudaPtr, numProxies * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(unsigned int);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaMemcpy(proxiesData + bufferOffset, cudaPtr, numProxies * sizeof(float), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(float);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceXys));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceXys));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceXys));
    CHECK_CUDA_ERROR(cudaMemcpy(proxiesData + bufferOffset, cudaPtr, numProxies * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(unsigned int);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceOffsetSizeFlatteneds));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceOffsetSizeFlatteneds));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceOffsetSizeFlatteneds));
    CHECK_CUDA_ERROR(cudaMemcpy(proxiesData + bufferOffset, cudaPtr, numProxies * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(unsigned int);
#else
    memcpy(proxiesData, &numProxies, sizeof(unsigned int));
    bufferOffset += sizeof(unsigned int);

    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.getSubData(0, numProxies, proxiesData + bufferOffset);
    bufferOffset += numProxies * sizeof(unsigned int);

    depthsBuffer.bind();
    depthsBuffer.getSubData(0, numProxies, proxiesData + bufferOffset);
    bufferOffset += numProxies * sizeof(float);

    xysBuffer.bind();
    xysBuffer.getSubData(0, numProxies, proxiesData + bufferOffset);
    bufferOffset += numProxies * sizeof(unsigned int);

    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.getSubData(0, numProxies, proxiesData + bufferOffset);
    bufferOffset += numProxies * sizeof(unsigned int);
#endif
    return bufferOffset;
}

unsigned int QuadBuffers::saveProxiesToFile(const std::string &filename) {
    unsigned int size = updateProxiesDataBuffer();
    std::ofstream quadsFile(filename, std::ios::binary);
    quadsFile.write(proxiesData, size);
    quadsFile.close();
    return size;
}
#endif
