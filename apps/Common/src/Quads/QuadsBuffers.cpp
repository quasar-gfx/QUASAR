#if !defined(__ANDROID__)
#include <filesystem>
#endif

#include <spdlog/spdlog.h>
#include <Utils/TimeUtils.h>

#include <Quads/QuadsBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(unsigned int maxProxies)
        : maxProxies(maxProxies)
        , numProxies(maxProxies)
        , normalSphericalsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(unsigned int), nullptr, GL_DYNAMIC_COPY)
        , depthsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(float), nullptr, GL_DYNAMIC_COPY)
        , offsetSizeFlattenedsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(unsigned int), nullptr, GL_DYNAMIC_COPY)
#if !defined(__APPLE__) && !defined(__ANDROID__)
        , cudaBufferNormalSphericals(normalSphericalsBuffer)
        , cudaBufferDepths(depthsBuffer)
        , cudaBufferOffsetSizeFlatteneds(offsetSizeFlattenedsBuffer)
#endif
        , data(sizeof(unsigned int) + maxProxies * sizeof(QuadMapDataPacked)) {

}

void QuadBuffers::resize(unsigned int numProxies) {
    this->numProxies = numProxies;
}

unsigned int QuadBuffers::loadFromMemory(const std::vector<char> &compressedData, bool decompress) {
    auto startTime = timeutils::getTimeMicros();
    if (decompress) {
        codec.decompress(compressedData, data);
    }
    else {
        data = compressedData;
    }
    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    unsigned int bufferOffset = 0;

    numProxies = *reinterpret_cast<const unsigned int*>(data.data());
    bufferOffset += sizeof(unsigned int);

    auto normalSphericalsPtr = reinterpret_cast<const unsigned int*>(data.data() + bufferOffset);
    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.setData(numProxies, normalSphericalsPtr);
    bufferOffset += numProxies * sizeof(unsigned int);

    auto depthsPtr = reinterpret_cast<const float*>(data.data() + bufferOffset);
    depthsBuffer.bind();
    depthsBuffer.setData(numProxies, depthsPtr);
    bufferOffset += numProxies * sizeof(float);

    auto offsetSizeFlattenedsPtr = reinterpret_cast<const unsigned int*>(data.data() + bufferOffset);
    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.setData(numProxies, offsetSizeFlattenedsPtr);

    return numProxies;
}

unsigned int QuadBuffers::loadFromFile(const std::string &filename, unsigned int* numBytesLoaded, bool compressed) {
#if !defined(__ANDROID__)
    if (!std::filesystem::exists(filename)) {
        spdlog::error("File {} does not exist", filename);
        return 0;
    }
#endif

    auto quadDataCompressed = FileIO::loadBinaryFile(filename, numBytesLoaded);

    auto startTime = timeutils::getTimeMicros();
    auto numBytes = loadFromMemory(quadDataCompressed, compressed);
    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    return numBytes;
}

#ifdef GL_CORE
unsigned int QuadBuffers::saveToMemory(std::vector<char> &compressedData, bool compress) {
    unsigned int dataSize = updateDataBuffer();
    compressedData.resize(dataSize);

    if (compress) {
        auto startTime = timeutils::getTimeMicros();
        unsigned int outputSize = codec.compress(data.data(), compressedData, dataSize);
        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return outputSize;
    }
    else {
        memcpy(compressedData.data(), data.data(), dataSize);
        return dataSize;
    }
}

unsigned int QuadBuffers::saveToFile(const std::string &filename) {
    std::vector<char> compressedData;
    unsigned int outputSize = saveToMemory(compressedData, true);

    std::ofstream quadsFile = std::ofstream(filename + ".zstd", std::ios::binary);
    quadsFile.write(compressedData.data(), outputSize);
    quadsFile.close();

    return outputSize;
}

unsigned int QuadBuffers::updateDataBuffer() {
    unsigned int bufferOffset = 0;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    void* cudaPtr;

    memcpy(data.data(), &numProxies, sizeof(unsigned int));
    bufferOffset += sizeof(unsigned int);

    cudaPtr = cudaBufferNormalSphericals.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(unsigned int);

    cudaPtr = cudaBufferDepths.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(float), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(float);

    cudaPtr = cudaBufferOffsetSizeFlatteneds.getPtr();
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
