#if !defined(__ANDROID__)
#include <filesystem>
#endif

#include <spdlog/spdlog.h>
#include <Utils/TimeUtils.h>

#include <Quads/QuadsBuffers.h>

QuadBuffers::QuadBuffers(unsigned int maxProxies)
        : maxProxies(maxProxies)
        , numProxies(maxProxies)
        , normalSphericalsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr)
        , depthsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr)
        , offsetSizeFlattenedsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxProxies, nullptr)
        , data(sizeof(unsigned int) + maxProxies * sizeof(QuadMapDataPacked)) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudautils::checkCudaDevice();
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceNormalSphericals, normalSphericalsBuffer, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceDepths, depthsBuffer, cudaGraphicsRegisterFlagsNone));
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceOffsetSizeFlatteneds, offsetSizeFlattenedsBuffer, cudaGraphicsRegisterFlagsNone));
#endif

    // setup LZ4 decompression context
    auto status = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
    if (LZ4F_isError(status)) {
        spdlog::error("Failed to create LZ4 context: {}", LZ4F_getErrorName(status));
        return;
    }
}

QuadBuffers::~QuadBuffers() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResourceOffsetSizeFlatteneds));
#endif

    LZ4F_freeDecompressionContext(dctx);
}

void QuadBuffers::resize(unsigned int numProxies) {
    this->numProxies = numProxies;
}

unsigned int QuadBuffers::loadFromFile(const std::string &filename, unsigned int* numBytesLoaded) {
#if !defined(__ANDROID__)
    if (!std::filesystem::exists(filename)) {
        spdlog::error("File {} does not exist", filename);
        return 0;
    }
#endif

    auto quadDataCompressed = FileIO::loadBinaryFile(filename);
    size_t srcSize = quadDataCompressed.size();
    size_t dstSize = maxProxies * sizeof(QuadMapDataPacked) + sizeof(unsigned int);

    if (numBytesLoaded != nullptr) {
        *numBytesLoaded = srcSize;
    }

    auto startTime = timeutils::getTimeMicros();
    auto ret = LZ4F_decompress(dctx,
                               data.data(), &dstSize,
                               quadDataCompressed.data(), &srcSize,
                               nullptr);
    if (LZ4F_isError(ret)) {
        spdlog::error("LZ4 decompression failed: {}", LZ4F_getErrorName(ret));
        return 0;
    }

    auto numBytes = loadFromMemory(reinterpret_cast<const char*>(data.data()));
    stats.timeToDecompressionMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    return numBytes;
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

    auto offsetSizeFlattenedsPtr = reinterpret_cast<const unsigned int*>(data + bufferOffset);
    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.setData(numProxies, offsetSizeFlattenedsPtr);

    return numProxies;
}

#ifdef GL_CORE
unsigned int QuadBuffers::saveToMemory(std::vector<char> &compressedData, bool doLZ4) {
    unsigned int dataSize = updateDataBuffer();

    if (doLZ4) {
        auto startTime = timeutils::getTimeMicros();

        int maxCompressedSize = LZ4_compressBound(dataSize);
        compressedData.resize(maxCompressedSize);

        int compressedSize = LZ4_compress_default(
            reinterpret_cast<const char*>(data.data()), // source data
            compressedData.data(),                      // destination buffer
            dataSize,                                   // input size
            maxCompressedSize                           // maximum output size
        );

        if (compressedSize <= 0) {
            spdlog::error("LZ4 compression failed.");
            return 0;
        }

        compressedData.resize(compressedSize);

        stats.timeToCompressionMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return compressedSize;
    }
    else {
        compressedData.resize(dataSize);
        memcpy(compressedData.data(), data.data(), dataSize);
        return dataSize;
    }
}

unsigned int QuadBuffers::saveToFile(const std::string &filename) {
    auto startTime = timeutils::getTimeMicros();

    std::vector<char> compressedData;
    unsigned int dataSize = saveToMemory(compressedData, true);

    std::ofstream quadsFile(filename + ".lz4", std::ios::binary);
    quadsFile.write(compressedData.data(), dataSize);
    quadsFile.close();

    stats.timeToCompressionMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    return dataSize;
}

unsigned int QuadBuffers::updateDataBuffer() {
    unsigned int bufferOffset = 0;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    void* cudaPtr;
    size_t size;

    memcpy(data.data(), &numProxies, sizeof(unsigned int));
    bufferOffset += sizeof(unsigned int);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceNormalSphericals));
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(unsigned int);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceDepths));
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(float), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(float);

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceOffsetSizeFlatteneds));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResourceOffsetSizeFlatteneds));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceOffsetSizeFlatteneds));
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(unsigned int);
#else
    memcpy(data.data(), &numProxies, sizeof(unsigned int));
    bufferOffset += sizeof(unsigned int);

    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(unsigned int);

    depthsBuffer.bind();
    depthsBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(float);

    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(unsigned int);
#endif
    return bufferOffset;
}
#endif
