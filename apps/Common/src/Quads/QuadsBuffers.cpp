#if !defined(__ANDROID__)
#include <filesystem>
#endif

#include <spdlog/spdlog.h>

#include <Quads/QuadsBuffers.h>
#include <Utils/TimeUtils.h>

using namespace quasar;

QuadBuffers::QuadBuffers(uint maxProxies)
        : maxProxies(maxProxies)
        , numProxies(maxProxies)
        , normalSphericalsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(uint), nullptr, GL_DYNAMIC_COPY)
        , depthsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(float), nullptr, GL_DYNAMIC_COPY)
        , offsetSizeFlattenedsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(uint), nullptr, GL_DYNAMIC_COPY)
#if !defined(__APPLE__) && !defined(__ANDROID__)
        , cudaBufferNormalSphericals(normalSphericalsBuffer)
        , cudaBufferDepths(depthsBuffer)
        , cudaBufferOffsetSizeFlatteneds(offsetSizeFlattenedsBuffer)
#endif
        , data(sizeof(uint) + maxProxies * sizeof(QuadMapDataPacked)) {

}

void QuadBuffers::resize(uint numProxies) {
    this->numProxies = numProxies;
}

uint QuadBuffers::loadFromMemory(const std::vector<char> &compressedData, bool decompress) {
    double startTime = timeutils::getTimeMicros();
    if (decompress) {
        codec.decompress(compressedData, data);
    }
    else {
        data = compressedData;
    }
    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

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

    auto offsetSizeFlattenedsPtr = reinterpret_cast<const uint*>(data.data() + bufferOffset);
    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.setData(numProxies, offsetSizeFlattenedsPtr);

    return numProxies;
}

uint QuadBuffers::loadFromFile(const std::string &filename, uint* numBytesLoaded, bool compressed) {
#if !defined(__ANDROID__)
    if (!std::filesystem::exists(filename)) {
        spdlog::error("File {} does not exist", filename);
        return 0;
    }
#endif

    auto quadDataCompressed = FileIO::loadBinaryFile(filename, numBytesLoaded);
    return loadFromMemory(quadDataCompressed, compressed);
}

#ifdef GL_CORE
uint QuadBuffers::saveToMemory(std::vector<char> &compressedData, bool compress) {
    uint dataSize = updateDataBuffer();
    compressedData.resize(dataSize);

    double startTime = timeutils::getTimeMicros();
    uint outputSize = dataSize;
    if (compress) {
        outputSize = codec.compress(data.data(), compressedData, dataSize);
        compressedData.resize(outputSize);
    }
    else {
        memcpy(compressedData.data(), data.data(), dataSize);
    }
    stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return outputSize;
}

uint QuadBuffers::saveToFile(const std::string &filename) {
    std::vector<char> compressedData;
    uint outputSize = saveToMemory(compressedData, true);

    std::ofstream quadsFile = std::ofstream(filename + ".zstd", std::ios::binary);
    quadsFile.write(compressedData.data(), outputSize);
    quadsFile.close();

    return outputSize;
}

uint QuadBuffers::updateDataBuffer() {
    uint bufferOffset = 0;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    void* cudaPtr;

    memcpy(data.data(), &numProxies, sizeof(uint));
    bufferOffset += sizeof(uint);

    cudaPtr = cudaBufferNormalSphericals.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(uint), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(uint);

    cudaPtr = cudaBufferDepths.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + bufferOffset, cudaPtr, numProxies * sizeof(float), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(float);

    cudaPtr = cudaBufferOffsetSizeFlatteneds.getPtr();
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

    offsetSizeFlattenedsBuffer.bind();
    offsetSizeFlattenedsBuffer.getSubData(0, numProxies, data.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint);
#endif
    return bufferOffset;
}
#endif
