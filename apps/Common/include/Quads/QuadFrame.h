#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

#include <Utils/FileIO.h>

#include <Codec/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class QuadFrame {
public:
    struct Stats {
        double timeToCompressMs = 0.0;
        double timeToDecompressMs = 0.0;
    } stats;

    // GPU buffers
    QuadBuffers quadBuffers;
    DepthOffsets depthOffsets;

    QuadFrame(const glm::uvec2& frameSize)
        : frameSize(frameSize)
        , quadBuffers(frameSize.x * frameSize.y)
        , depthOffsets(2u * frameSize) // 4 offsets per pixel
        , decompressedQuads(sizeof(uint) + quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
        , decompressedDepthOffsets(sizeof(uint) + depthOffsets.size.x * depthOffsets.size.y * 4 * sizeof(uint16_t))
    {}

    const std::vector<char>& getQuads() const {
        return compressedQuads;
    }

    const std::vector<char>& getDepthOffsets() const {
        return compressedDepthOffsets;
    }

    uint getNumQuads() const {
        return quadBuffers.numProxies;
    }

    uint getNumDepthOffsets() const {
        return depthOffsets.size.x * depthOffsets.size.y;
    }

    const glm::uvec2& getSize() const {
        return frameSize;
    }

    void setNumQuads(int numProxies) {
        copiedToCPU = false;
        quadBuffers.resize(numProxies);
    }

    std::pair<uint, uint> copyToMemory(std::vector<char>& outputQuads, std::vector<char>& outputDepthOffsets) const {
        outputQuads.resize(compressedQuads.size());
        outputDepthOffsets.resize(compressedDepthOffsets.size());

        std::copy(compressedQuads.begin(), compressedQuads.end(), outputQuads.begin());
        std::copy(compressedDepthOffsets.begin(), compressedDepthOffsets.end(), outputDepthOffsets.begin());

        return { outputQuads.size(), outputDepthOffsets.size() };
    }

#ifdef GL_CORE
    std::pair<uint, uint> copyToMemory() {
        if (copiedToCPU) {
            return { compressedQuads.size(), compressedDepthOffsets.size() };
        }

        double startTime = timeutils::getTimeMicros();

        // Copy quads
        quadBuffers.copyToMemory(decompressedQuads);
        uint savedQuadsSize = codec.compress(decompressedQuads.data(), compressedQuads, decompressedQuads.size());
        compressedQuads.resize(savedQuadsSize);

        // Copy depth offsets
        depthOffsets.copyToMemory(decompressedDepthOffsets);
        uint savedDepthOffsetsSize = codec.compress(decompressedDepthOffsets.data(), compressedDepthOffsets, decompressedDepthOffsets.size());
        compressedDepthOffsets.resize(savedDepthOffsetsSize);

        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return { savedQuadsSize, savedDepthOffsetsSize };
    }
#endif

    std::pair<uint, uint> loadFromFiles(const std::string& quadsFilename, const std::string& depthOffsetsFilename) {
        double startTime = timeutils::getTimeMicros();

        // Load quads
        uint bytesProxies;
        compressedQuads = FileIO::loadBinaryFile(quadsFilename, &bytesProxies);
        codec.decompress(compressedQuads, decompressedQuads);
        quadBuffers.loadFromMemory(decompressedQuads);

        // Load depth offsets
        uint bytesDepthOffsets;
        compressedDepthOffsets = FileIO::loadBinaryFile(depthOffsetsFilename, &bytesDepthOffsets);
        codec.decompress(compressedDepthOffsets, decompressedDepthOffsets);
        depthOffsets.loadFromMemory(decompressedDepthOffsets);

        stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        copiedToCPU = true;
        return { bytesProxies, bytesDepthOffsets };
    }

private:
    bool copiedToCPU = false;

    glm::uvec2 frameSize;

    ZSTDCodec codec;

    // CPU buffers
    std::vector<char> compressedQuads;
    std::vector<char> compressedDepthOffsets;

    std::vector<char> decompressedQuads;
    std::vector<char> decompressedDepthOffsets;
};

} // namespace quasar

#endif // QUAD_FRAME_H
