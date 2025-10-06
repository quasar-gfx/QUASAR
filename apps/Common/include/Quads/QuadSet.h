#ifndef QUAD_SET_H
#define QUAD_SET_H

#include <spdlog/spdlog.h>

#include <Path.h>
#include <Utils/FileIO.h>
#include <Codecs/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class QuadSet {
public:
    struct Sizes {
        size_t numQuads = 0;
        size_t numQuadsTransparent = 0;
        size_t numDepthOffsets = 0;
        double quadsSize = 0.0;
        double depthOffsetsSize = 0.0;

        Sizes operator+(const Sizes& other) const {
            return {
                numQuads + other.numQuads,
                numQuadsTransparent + other.numQuadsTransparent,
                numDepthOffsets + other.numDepthOffsets,
                quadsSize + other.quadsSize,
                depthOffsetsSize + other.depthOffsetsSize
            };
        }
        Sizes& operator+=(const Sizes& other) {
            numQuads += other.numQuads;
            numQuadsTransparent += other.numQuadsTransparent;
            numDepthOffsets += other.numDepthOffsets;
            quadsSize += other.quadsSize;
            depthOffsetsSize += other.depthOffsetsSize;
            return *this;
        }
    };

    struct Stats {
        double transferTimeMs = 0.0;
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

    size_t getNumProxies() const {
        return quadBuffers.numProxies;
    }

    size_t getNumProxiesTransparent() const {
        return quadBuffers.numProxiesTransparent;
    }

    size_t getNumDepthOffsets() const {
        return depthOffsets.getSize().x * depthOffsets.getSize().y;
    }

    void setNumProxies(int numProxies, int numProxiesTransparent) {
        quadBuffers.resize(numProxies, numProxiesTransparent);
    }

    Sizes writeToMemory(std::vector<char>& outputQuads, std::vector<char>& outputDepthOffsets, bool applyDeltaEncoding = true) {
#ifdef GL_CORE
        double startTime = timeutils::getTimeMicros();
        quadBuffers.writeToMemory(outputQuads, applyDeltaEncoding);
        depthOffsets.writeToMemory(outputDepthOffsets);
        stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
#else
        spdlog::error("QuadSet::writeToMemory is only supported in OpenGL Core");
#endif

        return {
            quadBuffers.numProxies,
            quadBuffers.numProxiesTransparent,
            depthOffsets.getSize().x * depthOffsets.getSize().y,
            static_cast<double>(outputQuads.size()),
            static_cast<double>(outputDepthOffsets.size()),
        };
    }

    Sizes loadFromMemory(std::vector<char>& inputQuads, std::vector<char>& inputDepthOffsets, bool applyDeltaEncoding = true) {
        double startTime = timeutils::getTimeMicros();
        auto quadsSize = quadBuffers.loadFromMemory(inputQuads, applyDeltaEncoding);
        depthOffsets.loadFromMemory(inputDepthOffsets);
        stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return {
            quadBuffers.numProxies,
            quadBuffers.numProxiesTransparent,
            depthOffsets.getSize().x * depthOffsets.getSize().y,
            static_cast<double>(quadsSize),
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
