#ifndef QUAD_SET_H
#define QUAD_SET_H

#include <Path.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class QuadSet {
public:
    struct Sizes {
        size_t numQuads = 0;
        size_t numDepthOffsets = 0;
        double quadsSize = 0.0;
        double depthOffsetsSize = 0.0;

        Sizes& operator+=(const Sizes& other) {
            numQuads += other.numQuads;
            numDepthOffsets += other.numDepthOffsets;
            quadsSize += other.quadsSize;
            depthOffsetsSize += other.depthOffsetsSize;
            return *this;
        }
    };

    struct Stats {
        double timeToTransferMs = 0.0;
    } stats;

    QuadBuffers quadBuffers;
    DepthOffsets depthOffsets;

    QuadSet(const glm::uvec2& frameSize)
        : frameSize(frameSize)
        , quadBuffers(frameSize.x * frameSize.y)
        , depthOffsets(2u * frameSize) // 2x2 subpixels per pixel
    {}

    const glm::uvec2& getSize() const {
        return frameSize;
    }

    uint getNumQuads() const {
        return quadBuffers.numProxies;
    }

    uint getNumDepthOffsets() const {
        return depthOffsets.getSize().x * depthOffsets.getSize().y;
    }

    void setNumQuads(int numProxies) {
        quadBuffers.resize(numProxies);
    }

#ifdef GL_CORE
    Sizes copyToCPU(std::vector<char>& outputQuads, std::vector<char>& outputDepthOffsets) {
        double startTime = timeutils::getTimeMicros();
        quadBuffers.copyToCPU(outputQuads);
        depthOffsets.copyToCPU(outputDepthOffsets);
        stats.timeToTransferMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return {
            quadBuffers.numProxies,
            depthOffsets.getSize().x * depthOffsets.getSize().y,
            static_cast<double>(outputQuads.size()),
            static_cast<double>(outputDepthOffsets.size()),
        };
    }
#endif

    Sizes copyFromCPU(std::vector<char>& inputQuads, std::vector<char>& inputDepthOffsets) {
        double startTime = timeutils::getTimeMicros();
        quadBuffers.copyFromCPU(inputQuads);
        depthOffsets.copyFromCPU(inputDepthOffsets);
        stats.timeToTransferMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return {
            quadBuffers.numProxies,
            depthOffsets.getSize().x * depthOffsets.getSize().y,
            static_cast<double>(inputQuads.size()),
            static_cast<double>(inputDepthOffsets.size())
        };
    }

private:
    glm::uvec2 frameSize;

    ZSTDCodec quadCodec;
    ZSTDCodec depthOffsetCodec;
};

} // namespace quasar

#endif // QUAD_SET_H
