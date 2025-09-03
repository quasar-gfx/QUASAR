#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

#include <algorithm>

#include <spdlog/spdlog.h>

#include <Path.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class QuadFrame {
public:
    enum FrameType {
        NONE,
        REFERENCE,
        RESIDUAL,
    };

    QuadFrame(FrameType type) : frameType(type) {}
    virtual ~QuadFrame() = default;

    FrameType getFrameType() const { return frameType; }

protected:
    FrameType frameType;
};

class ReferenceFrame : public QuadFrame {
public:
    struct Header {
        uint32_t quadsSize;
        uint32_t depthOffsetsSize;
    };

    size_t numQuads, numDepthOffsets;
    std::vector<char> quads, depthOffsets;

    ReferenceFrame()
        : QuadFrame(FrameType::REFERENCE)
        , numQuads(0)
        , numDepthOffsets(0)
    {}

    size_t getTotalNumQuads() const { return numQuads; }
    size_t getTotalNumDepthOffsets() const { return numDepthOffsets; }
    double getTotalQuadsSize() const { return quads.size(); }
    double getTotalDepthOffsetsSize() const { return depthOffsets.size(); }

    size_t compressAndStoreQuads(const std::vector<char>& uncompressedQuads) {
        return refQuadsCodec.compress(
            uncompressedQuads.data(),
            quads,
            uncompressedQuads.size());
    }
    size_t compressAndStoreDepthOffsets(const std::vector<char>& uncompressedOffsets) {
        return refOffsetsCodec.compress(
            uncompressedOffsets.data(),
            depthOffsets,
            uncompressedOffsets.size());
    }

    size_t decompressQuads(std::vector<char>& outputQuads) {
        return refQuadsCodec.decompress(quads, outputQuads);
    }
    size_t decompressDepthOffsets(std::vector<char>& outputOffsets) {
        return refOffsetsCodec.decompress(depthOffsets, outputOffsets);
    }

    double getTimeToCompress() const {
        return std::max(refQuadsCodec.stats.timeToCompressMs, refOffsetsCodec.stats.timeToCompressMs);
    }
    double getTimeToDecompress() const {
        return std::max(refQuadsCodec.stats.timeToDecompressMs, refOffsetsCodec.stats.timeToDecompressMs);
    }

    size_t writeToFiles(const Path& outputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Save quads
        double startTime = timeutils::getTimeMicros();
        Path quadsFileName = (outputPath / ("quads" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(quadsFileName, quads.data(), quads.size());
        spdlog::debug("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                       numQuads, static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save depth offsets
        startTime = timeutils::getTimeMicros();
        Path depthOffsetsFileName = (outputPath / ("depthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(depthOffsetsFileName, depthOffsets.data(), depthOffsets.size());
        spdlog::debug("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsets, static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }

    size_t writeToMemory(std::vector<char>& outputData) {
        Header header{
            static_cast<uint32_t>(quads.size()),
            static_cast<uint32_t>(depthOffsets.size())
        };
        size_t outputSize = sizeof(header) + quads.size() + depthOffsets.size();
        outputData.resize(outputSize);

        char* ptr = outputData.data();
        // Write header
        std::memcpy(ptr, &header, sizeof(Header));
        ptr += sizeof(Header);
        // Write quads
        std::memcpy(ptr, quads.data(), quads.size());
        ptr += quads.size();
        // Write depth offsets
        std::memcpy(ptr, depthOffsets.data(), depthOffsets.size());
        ptr += depthOffsets.size();

        return outputSize;
    }

    size_t loadFromFiles(const Path& inputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Load quads
        double startTime = timeutils::getTimeMicros();
        Path quadsFileName = (inputPath / ("quads" + idxStr)).withExtension(".bin.zstd");
        const auto& quadsData = FileIO::loadFromBinaryFile(quadsFileName);
        quads = std::move(quadsData);
        spdlog::debug("Loaded quads ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        startTime = timeutils::getTimeMicros();
        Path depthOffsetsFileName = (inputPath / ("depthOffsets" + idxStr)).withExtension(".bin.zstd");
        const auto& depthOffsetsData = FileIO::loadFromBinaryFile(depthOffsetsFileName);
        depthOffsets = std::move(depthOffsetsData);
        spdlog::debug("Loaded depth offsets ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }

    size_t loadFromMemory(const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);

        // Read header
        Header header;
        std::memcpy(&header, ptr, sizeof(Header));
        ptr += sizeof(Header);

        // Sanity check
        size_t expectedSize = sizeof(Header) + header.quadsSize + header.depthOffsetsSize;
        if (size < expectedSize) {
            throw std::runtime_error("Reference Frame input data size " +
                                     std::to_string(size) +
                                     " is smaller than expected from header " +
                                     std::to_string(expectedSize));
        }

        // Read quads
        quads.resize(header.quadsSize);
        std::memcpy(quads.data(), ptr, header.quadsSize);
        ptr += header.quadsSize;

        // Read depth offsets
        depthOffsets.resize(header.depthOffsetsSize);
        std::memcpy(depthOffsets.data(), ptr, header.depthOffsetsSize);
        ptr += header.depthOffsetsSize;

        return expectedSize;
    }

private:
    ZSTDCodec refQuadsCodec;
    ZSTDCodec refOffsetsCodec;
};

class ResidualFrame : public QuadFrame {
public:
    struct Header {
        uint32_t quadsUpdatedSize;
        uint32_t depthOffsetsUpdatedSize;
        uint32_t quadsRevealedSize;
        uint32_t depthOffsetsRevealedSize;
    };

    size_t numQuadsUpdated;
    size_t numDepthOffsetsUpdated;
    size_t numQuadsRevealed;
    size_t numDepthOffsetsRevealed;
    std::vector<char> quadsUpdated;
    std::vector<char> depthOffsetsUpdated;
    std::vector<char> quadsRevealed;
    std::vector<char> depthOffsetsRevealed;

    ResidualFrame()
        : QuadFrame(FrameType::RESIDUAL)
        , numQuadsUpdated(0)
        , numDepthOffsetsUpdated(0)
        , numQuadsRevealed(0)
        , numDepthOffsetsRevealed(0)
    {}

    size_t getTotalNumQuads() const { return numQuadsUpdated + numQuadsRevealed; }
    size_t getTotalNumDepthOffsets() const { return numDepthOffsetsUpdated + numDepthOffsetsRevealed; }
    size_t getTotalNumQuadsUpdated() const { return numQuadsUpdated; }
    size_t getTotalNumQuadsRevealed() const { return numQuadsRevealed; }
    double getTotalQuadsSize() const { return quadsUpdated.size() + depthOffsetsUpdated.size(); }
    double getTotalDepthOffsetsSize() const { return quadsRevealed.size() + depthOffsetsRevealed.size(); }

    size_t compressAndStoreUpdatedQuads(const std::vector<char>& uncompressedQuads) {
        return resQuadsUpdatedCodec.compress(
            uncompressedQuads.data(),
            quadsUpdated,
            uncompressedQuads.size());
    }
    size_t compressAndStoreRevealedQuads(const std::vector<char>& uncompressedQuads) {
        return resQuadsRevealedCodec.compress(
            uncompressedQuads.data(),
            quadsRevealed,
            uncompressedQuads.size());
    }
    size_t compressAndStoreUpdatedDepthOffsets(const std::vector<char>& uncompressedOffsets) {
        return resOffsetsUpdatedCodec.compress(
            uncompressedOffsets.data(),
            depthOffsetsUpdated,
            uncompressedOffsets.size());
    }
    size_t compressAndStoreRevealedDepthOffsets(const std::vector<char>& uncompressedOffsets) {
        return resOffsetsRevealedCodec.compress(
            uncompressedOffsets.data(),
            depthOffsetsRevealed,
            uncompressedOffsets.size());
    }

    size_t decompressUpdatedQuads(std::vector<char>& outputQuads) {
        return resQuadsUpdatedCodec.decompress(quadsUpdated, outputQuads);
    }
    size_t decompressRevealedQuads(std::vector<char>& outputQuads) {
        return resQuadsRevealedCodec.decompress(quadsRevealed, outputQuads);
    }
    size_t decompressUpdatedDepthOffsets(std::vector<char>& outputOffsets) {
        return resOffsetsUpdatedCodec.decompress(depthOffsetsUpdated, outputOffsets);
    }
    size_t decompressRevealedDepthOffsets(std::vector<char>& outputOffsets) {
        return resOffsetsRevealedCodec.decompress(depthOffsetsRevealed, outputOffsets);
    }

    double getTimeToCompress() const {
        return std::max(
            std::max(resQuadsUpdatedCodec.stats.timeToCompressMs, resOffsetsUpdatedCodec.stats.timeToCompressMs),
            std::max(resQuadsRevealedCodec.stats.timeToCompressMs, resOffsetsRevealedCodec.stats.timeToCompressMs)
        );
    }
    double getTimeToDecompress() const {
        return std::max(
            std::max(resQuadsUpdatedCodec.stats.timeToDecompressMs, resOffsetsUpdatedCodec.stats.timeToDecompressMs),
            std::max(resQuadsRevealedCodec.stats.timeToDecompressMs, resOffsetsRevealedCodec.stats.timeToDecompressMs)
        );
    }

    size_t writeToFiles(const Path& outputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Save updated quads
        double startTime = timeutils::getTimeMicros();
        Path updatedQuadsFileName = (outputPath / ("updatedQuads" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(updatedQuadsFileName, quadsUpdated.data(), quadsUpdated.size());
        spdlog::debug("Saved {} updated quads ({:.3f}MB) in {:.3f}ms",
                       numQuadsUpdated, static_cast<double>(quadsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save updated depth offsets
        startTime = timeutils::getTimeMicros();
        Path updatedDepthOffsetsFileName = (outputPath / ("updatedDepthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(updatedDepthOffsetsFileName, depthOffsetsUpdated.data(), depthOffsetsUpdated.size());
        spdlog::debug("Saved {} updated depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsUpdated, static_cast<double>(depthOffsetsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed quads
        startTime = timeutils::getTimeMicros();
        Path revealedQuadsFileName = (outputPath / ("revealedQuads" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(revealedQuadsFileName, quadsRevealed.data(), quadsRevealed.size());
        spdlog::debug("Saved {} revealed quads ({:.3f}MB) in {:.3f}ms",
                        numQuadsRevealed, static_cast<double>(quadsRevealed.size()) / BYTES_PER_MEGABYTE,
                          timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed depth offsets
        startTime = timeutils::getTimeMicros();
        Path revealedDepthOffsetsFileName = (outputPath / ("revealedDepthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(revealedDepthOffsetsFileName, depthOffsetsRevealed.data(), depthOffsetsRevealed.size());
        spdlog::debug("Saved {} revealed depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsRevealed, static_cast<double>(depthOffsetsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quadsUpdated.size() + depthOffsetsUpdated.size() +
               quadsRevealed.size() + depthOffsetsRevealed.size();
    }

    size_t loadFromFiles(const Path& inputPath) {
        // Load updated quads
        double startTime = timeutils::getTimeMicros();
        Path updatedQuadsFileName = (inputPath / "updatedQuads").withExtension(".bin.zstd");
        const auto& quadsUpdatedData = FileIO::loadFromBinaryFile(updatedQuadsFileName);
        quadsUpdated = std::move(quadsUpdatedData);
        spdlog::debug("Loaded updated quads ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(quadsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Load updated depth offsets
        startTime = timeutils::getTimeMicros();
        Path updatedDepthOffsetsFileName = (inputPath / "updatedDepthOffsets").withExtension(".bin.zstd");
        const auto& depthOffsetsUpdatedData = FileIO::loadFromBinaryFile(updatedDepthOffsetsFileName);
        depthOffsetsUpdated = std::move(depthOffsetsUpdatedData);
        spdlog::debug("Loaded updated depth offsets ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(depthOffsetsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Load revealed quads
        startTime = timeutils::getTimeMicros();
        Path revealedQuadsFileName = (inputPath / "revealedQuads").withExtension(".bin.zstd");
        const auto& quadsRevealedData = FileIO::loadFromBinaryFile(revealedQuadsFileName);
        quadsRevealed = std::move(quadsRevealedData);
        spdlog::debug("Loaded revealed quads ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(quadsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Load revealed depth offsets
        startTime = timeutils::getTimeMicros();
        Path revealedDepthOffsetsFileName = (inputPath / "revealedDepthOffsets").withExtension(".bin.zstd");
        const auto& depthOffsetsRevealedData = FileIO::loadFromBinaryFile(revealedDepthOffsetsFileName);
        depthOffsetsRevealed = std::move(depthOffsetsRevealedData);
        spdlog::debug("Loaded revealed depth offsets ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(depthOffsetsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quadsUpdated.size() + depthOffsetsUpdated.size() +
               quadsRevealed.size() + depthOffsetsRevealed.size();
    }

    size_t writeToMemory(std::vector<char>& outputData) {
        Header header{
            static_cast<uint32_t>(quadsUpdated.size()),
            static_cast<uint32_t>(depthOffsetsUpdated.size()),
            static_cast<uint32_t>(quadsRevealed.size()),
            static_cast<uint32_t>(depthOffsetsRevealed.size())
        };
        size_t outputSize = sizeof(Header) +
                            quadsUpdated.size() + depthOffsetsUpdated.size() +
                            quadsRevealed.size() + depthOffsetsRevealed.size();
        outputData.resize(outputSize);

        char* ptr = outputData.data();
        // Write header
        std::memcpy(ptr, &header, sizeof(Header));
        ptr += sizeof(Header);
        // Write updated quads
        std::memcpy(ptr, quadsUpdated.data(), quadsUpdated.size());
        ptr += quadsUpdated.size();
        // Write updated depth offsets
        std::memcpy(ptr, depthOffsetsUpdated.data(), depthOffsetsUpdated.size());
        ptr += depthOffsetsUpdated.size();
        // Write revealed quads
        std::memcpy(ptr, quadsRevealed.data(), quadsRevealed.size());
        ptr += quadsRevealed.size();
        // Write revealed depth offsets
        std::memcpy(ptr, depthOffsetsRevealed.data(), depthOffsetsRevealed.size());
        ptr += depthOffsetsRevealed.size();

        return outputData.size();
    }

    size_t loadFromMemory(const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);

        // Read header
        Header header;
        std::memcpy(&header, ptr, sizeof(Header));
        ptr += sizeof(Header);

        // Sanity check
        size_t expectedSize = sizeof(Header) +
                              header.quadsUpdatedSize + header.depthOffsetsUpdatedSize +
                              header.quadsRevealedSize + header.depthOffsetsRevealedSize;
        if (size < expectedSize) {
            throw std::runtime_error("Residual Frame input data size " +
                                     std::to_string(size) +
                                     " is smaller than expected from header " +
                                     std::to_string(expectedSize));
        }

        // Read updated quads
        quadsUpdated.resize(header.quadsUpdatedSize);
        std::memcpy(quadsUpdated.data(), ptr, header.quadsUpdatedSize);
        ptr += header.quadsUpdatedSize;

        // Read updated depth offsets
        depthOffsetsUpdated.resize(header.depthOffsetsUpdatedSize);
        std::memcpy(depthOffsetsUpdated.data(), ptr, header.depthOffsetsUpdatedSize);
        ptr += header.depthOffsetsUpdatedSize;

        // Read revealed quads
        quadsRevealed.resize(header.quadsRevealedSize);
        std::memcpy(quadsRevealed.data(), ptr, header.quadsRevealedSize);
        ptr += header.quadsRevealedSize;

        // Read revealed depth offsets
        depthOffsetsRevealed.resize(header.depthOffsetsRevealedSize);
        std::memcpy(depthOffsetsRevealed.data(), ptr, header.depthOffsetsRevealedSize);
        ptr += header.depthOffsetsRevealedSize;

        return expectedSize;
    }

private:
    ZSTDCodec resQuadsUpdatedCodec;
    ZSTDCodec resOffsetsUpdatedCodec;
    ZSTDCodec resQuadsRevealedCodec;
    ZSTDCodec resOffsetsRevealedCodec;
};

} // namespace quasar

#endif // QUAD_FRAME_H
