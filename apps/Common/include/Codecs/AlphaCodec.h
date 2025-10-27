#ifndef ALPHA_CODEC_H
#define ALPHA_CODEC_H

#include <algorithm>
#include <cstdint>
#include <vector>
#include <Codecs/Codec.h>
#include <Codecs/ZSTDCodec.h>

namespace quasar {

class AlphaCodec : public Codec {
public:
    struct Stats {
        double compressTimeMs = 0.0;
        double decompressTimeMs = 0.0;
    } stats;

    AlphaCodec(uint width, uint height)
        : width(width)
        , height(height)
    {
        prevFrame.resize(width * height, 0);
    }
    ~AlphaCodec() = default;

    size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) {
        double start = timeutils::getTimeMicros();

        const uint8_t* src = static_cast<const uint8_t*>(uncompressedData);

        // Bit-packed skip flags: 1 bit per block
        const uint nbx = blocksX(), nby = blocksY();
        const uint numBlocks = nbx * nby;
        const size_t flagBytes = (numBlocks + 7u) >> 3;

        deltaBuffer.clear();
        deltaBuffer.resize(flagBytes, 0); // placeholder for flags
        size_t writePos = flagBytes;

        // Build flags and residuals
        uint blockIndex = 0;
        for (uint by = 0; by < nby; by++) {
            for (uint bx = 0; bx < nbx; bx++, blockIndex++) {
                const BlockInfo bi = blockInfo(bx, by);
                const uint32_t sad = blockSumAbsDiff(src, bi);
                const bool skip = (sad <= sadThreshold);

                // If sum of absolute differences is below threshold, mark block as skipped
                if (skip) deltaBuffer[blockIndex >> 3] |= char(1u << (blockIndex & 7u));

                if (!skip) {
                    // Emit residuals for this block
                    iterateBlock(bi, [&](size_t idx, uint8_t v) {
                        const int16_t r = int16_t(v) - int16_t(prevFrame[idx]);
                        prevFrame[idx] = v;
                        if (writePos == deltaBuffer.size()) deltaBuffer.push_back(0);
                        deltaBuffer[writePos++] = static_cast<char>(static_cast<int8_t>(r));
                    });
                }
                else {
                    // For skipped blocks, just advance prevFrame to src (copy)
                    iterateBlock(bi, [&](size_t idx, uint8_t v) {
                        prevFrame[idx] = v;
                    });
                }
            }
        }

        // Compress
        size_t outputSize = zstd.compress(deltaBuffer.data(), compressedData, writePos);
        compressedData.resize(outputSize);

        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - start);
        return outputSize;
    }

    size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) {
        double start = timeutils::getTimeMicros();

        const uint nbx = blocksX(), nby = blocksY();
        const uint numBlocks = nbx * nby;
        const size_t flagBytes = (numBlocks + 7u) >> 3;

        // Decompress
        size_t decompressedBytes = zstd.decompress(compressedData, deltaBuffer, numBytesCompressed);
        if (decompressedBytes < flagBytes) {
            spdlog::error("AlphaCodec: decompression failed, decompressed data is too small");
            return 0;
        }

        const char* in = deltaBuffer.data();
        const char* inEnd = deltaBuffer.data() + decompressedBytes;

        // Flags
        const uint8_t* flags = reinterpret_cast<const uint8_t*>(in);
        const char* payload = in + flagBytes;

        size_t readPos = 0;
        uint blockIndex = 0;
        for (uint by = 0; by < nby; by++) {
            for (uint bx = 0; bx < nbx; bx++, blockIndex++) {
                const BlockInfo bi = blockInfo(bx, by);
                const bool skip = ((flags[blockIndex >> 3] >> (blockIndex & 7u)) & 1u) != 0;

                if (skip) {
                    // Copy prevFrame into output, keep prevFrame as-is (already prev)
                    iterateBlock(bi, [&](size_t idx, uint8_t) {
                        const uint8_t v = prevFrame[idx];
                        decompressedData[idx] = static_cast<char>(v);
                    });
                }
                else {
                    // Reconstruct from residuals
                    iterateBlock(bi, [&](size_t idx, uint8_t) {
                        if (payload + readPos >= inEnd) return; // safety
                        const int8_t d = static_cast<int8_t>(deltaBuffer[flagBytes + readPos++]);
                        const uint8_t v = uint8_t(int16_t(prevFrame[idx]) + d);
                        prevFrame[idx] = v;
                        decompressedData[idx] = static_cast<char>(v);
                    });
                }
            }
        }

        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - start);
        return decompressedBytes;
    }

private:
    uint width, height;
    static constexpr uint blockWidth = 8;
    static constexpr uint blockHeight = 8;

    // Skip tiles whose block SAD <= threshold
    static constexpr uint32_t sadThreshold = 0;

    ZSTDCodec zstd;
    std::vector<char> deltaBuffer;  // [flags | residuals]
    std::vector<uint8_t> prevFrame; // last decoded/encoded frame

    inline uint blocksX() const { return (width  + blockWidth  - 1) / blockWidth; }
    inline uint blocksY() const { return (height + blockHeight - 1) / blockHeight; }

    struct BlockInfo { uint x0, y0, w, h; };
    inline BlockInfo blockInfo(uint bx, uint by) const {
        const uint x0 = bx * blockWidth;
        const uint y0 = by * blockHeight;
        const uint w = std::min<uint>(blockWidth,  width  - x0);
        const uint h = std::min<uint>(blockHeight, height - y0);
        return { x0, y0, w, h };
    }

    template <typename F>
    inline void iterateBlock(const BlockInfo& b, F f) const {
        for (uint dy = 0; dy < b.h; dy++) {
            const size_t row = size_t(b.y0 + dy) * width + b.x0;
            for (uint dx = 0; dx < b.w; dx++) {
                const size_t idx = row + dx;
                f(idx, 0);
            }
        }
    }

    inline uint32_t blockSumAbsDiff(const uint8_t* src, const BlockInfo& b) const {
        uint32_t sad = 0;
        for (uint dy = 0; dy < b.h; dy++) {
            const size_t row = (b.y0 + dy) * width + b.x0;
            for (uint dx = 0; dx < b.w; dx++) {
                const size_t idx = row + dx;
                sad += uint32_t(std::abs(int(src[idx]) - int(prevFrame[idx])));
            }
        }
        return sad;
    }
};

} // namespace quasar

#endif // ALPHA_CODEC_H
