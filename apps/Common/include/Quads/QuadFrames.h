#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

#include <spdlog/spdlog.h>

#include <Path.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class ReferenceFrame {
public:
    size_t numQuads;
    size_t numDepthOffsets;
    std::vector<char> quads;
    std::vector<char> depthOffsets;

    size_t getTotalNumQuads() const {
        return numQuads;
    }
    size_t getTotalNumDepthOffsets() const {
        return numDepthOffsets;
    }
    double getTotalQuadsSize() const {
        return quads.size();
    }
    double getTotalDepthOffsetsSize() const {
        return depthOffsets.size();
    }

    uint saveToFiles(const Path& outputPath) {
        // Save quads
        double startTime = timeutils::getTimeMicros();
        Path filename = (outputPath / "quads").withExtension(".bin.zstd");
        std::ofstream quadsFile = std::ofstream(filename, std::ios::binary);
        quadsFile.write(quads.data(), quads.size());
        quadsFile.close();
        spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                       numQuads, static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save depth offsets
        startTime = timeutils::getTimeMicros();
        Path offsetsFile = (outputPath / "depthOffsets").withExtension(".bin.zstd");
        std::ofstream depthOffsetsFile = std::ofstream(offsetsFile, std::ios::binary);
        depthOffsetsFile.write(depthOffsets.data(), depthOffsets.size());
        depthOffsetsFile.close();
        spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsets, static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }
};

class ResidualFrame {
public:
    size_t numQuadsUpdated;
    size_t numDepthOffsetsUpdated;
    size_t numQuadsRevealed;
    size_t numDepthOffsetsRevealed;
    std::vector<char> quadsUpdated;
    std::vector<char> depthOffsetsUpdated;
    std::vector<char> quadsRevealed;
    std::vector<char> depthOffsetsRevealed;

    size_t getTotalNumQuads() const {
        return numQuadsUpdated + numQuadsRevealed;
    }
    size_t getTotalNumDepthOffsets() const {
        return numDepthOffsetsUpdated + numDepthOffsetsRevealed;
    }
    double getTotalQuadsSize() const {
        return quadsUpdated.size() + depthOffsetsUpdated.size();
    }
    double getTotalDepthOffsetsSize() const {
        return quadsRevealed.size() + depthOffsetsRevealed.size();
    }

    uint saveToFiles(const Path& outputPath) {
        // Save updated quads
        double startTime = timeutils::getTimeMicros();
        Path filename = (outputPath / "quads").withExtension(".bin.zstd");
        std::ofstream quadsFile = std::ofstream(filename, std::ios::binary);
        quadsFile.write(quadsUpdated.data(), quadsUpdated.size());
        quadsFile.close();
        spdlog::info("Saved {} updated quads ({:.3f}MB) in {:.3f}ms",
                       numQuadsUpdated, static_cast<double>(quadsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save depth offsets
        startTime = timeutils::getTimeMicros();
        Path offsetsFile = (outputPath / "depthOffsets").withExtension(".bin.zstd");
        std::ofstream depthOffsetsFile = std::ofstream(offsetsFile, std::ios::binary);
        depthOffsetsFile.write(depthOffsetsUpdated.data(), depthOffsetsUpdated.size());
        depthOffsetsFile.close();
        spdlog::info("Saved {} updated depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsUpdated, static_cast<double>(depthOffsetsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed quads
        startTime = timeutils::getTimeMicros();
        Path revealedFile = (outputPath / "revealedQuads").withExtension(".bin.zstd");
        std::ofstream revealedQuadsFile = std::ofstream(revealedFile, std::ios::binary);
        revealedQuadsFile.write(quadsRevealed.data(), quadsRevealed.size());
        revealedQuadsFile.close();
        spdlog::info("Saved {} revealed quads ({:.3f}MB) in {:.3f}ms",
                        numQuadsRevealed, static_cast<double>(quadsRevealed.size()) / BYTES_PER_MEGABYTE,
                          timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quadsUpdated.size() + depthOffsetsUpdated.size() +
               quadsRevealed.size() + depthOffsetsRevealed.size();
    }
};

} // namespace quasar

#endif // QUAD_FRAME_H
