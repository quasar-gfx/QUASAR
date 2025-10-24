#ifndef ALPHA_CODEC_H
#define ALPHA_CODEC_H

#include <Codecs/Codec.h>
#include <Codecs/ZSTDCodec.h>
#include <algorithm>

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
    {}
    ~AlphaCodec() = default;

    size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) {
        double startTime = timeutils::getTimeMicros();

        const uint8_t* src = static_cast<const uint8_t*>(uncompressedData);

        const size_t targetSize = static_cast<size_t>(width) * static_cast<size_t>(height);
        deltaBuffer.resize(targetSize);

        // Process image in 8x8 blocks.
        const uint blocksX = (width + blockWidth - 1) / blockWidth;
        const uint blocksY = (height + blockHeight - 1) / blockHeight;

        size_t outPos = 0;
        bool first = true;
        uint8_t prev = 0;
        for (uint by = 0; by < blocksY; by++) {
            for (uint bx = 0; bx < blocksX; bx++) {
                // Iterate rows inside block
                for (uint ry = 0; ry < blockHeight; ry++) {
                    const size_t y = static_cast<size_t>(by) * blockHeight + ry;
                    if (y >= height) break; // partial block at bottom
                    const size_t rowBase = y * static_cast<size_t>(width);
                    for (uint rx = 0; rx < blockWidth; rx++) {
                        const size_t x = static_cast<size_t>(bx) * blockWidth + rx;
                        if (x >= width) break; // partial block at right edge
                        const uint8_t v = src[rowBase + x];
                        if (first) {
                            deltaBuffer[outPos++] = static_cast<char>(v);
                            prev = v;
                            first = false;
                        }
                        else {
                            uint8_t d = static_cast<uint8_t>(v - prev);
                            deltaBuffer[outPos++] = static_cast<char>(d);
                            prev = v;
                        }
                    }
                }
            }
        }

        // Compress only the used portion
        size_t outputSize = zstd.compress(deltaBuffer.data(), compressedData, outPos);
        compressedData.resize(outputSize);

        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return outputSize;
    }

    size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) {
        double startTime = timeutils::getTimeMicros();

        const size_t targetSize = static_cast<size_t>(width) * static_cast<size_t>(height);
        deltaBuffer.resize(targetSize);

        size_t decompressedBytes = zstd.decompress(compressedData, deltaBuffer, numBytesCompressed);

        const uint blocksX = (width + blockWidth - 1) / blockWidth;
        const uint blocksY = (height + blockHeight - 1) / blockHeight;

        // Reconstruct pixels in the same 8x8 block order used at compression
        size_t inPos = 0;
        bool first = true;
        uint8_t prev = 0;
        for (uint by = 0; by < blocksY && inPos < decompressedBytes; by++) {
            for (uint bx = 0; bx < blocksX && inPos < decompressedBytes; bx++) {
                for (uint ry = 0; ry < blockHeight && inPos < decompressedBytes; ry++) {
                    const size_t y = static_cast<size_t>(by) * blockHeight + ry;
                    if (y >= height) break;
                    const size_t rowBase = y * static_cast<size_t>(width);
                    for (uint rx = 0; rx < blockWidth && inPos < decompressedBytes; rx++) {
                        const size_t x = static_cast<size_t>(bx) * blockWidth + rx;
                        if (x >= width) break;
                        uint8_t val;
                        if (first) {
                            val = static_cast<uint8_t>(deltaBuffer[inPos++]);
                            prev = val;
                            first = false;
                        }
                        else {
                            uint8_t d = static_cast<uint8_t>(deltaBuffer[inPos++]);
                            val = static_cast<uint8_t>(prev + d);
                            prev = val;
                        }
                        decompressedData[rowBase + x] = static_cast<char>(val);
                    }
                }
            }
        }

        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        size_t outputSize = std::min(decompressedBytes, targetSize);
        return outputSize;
    }

private:
    uint width, height;

    const uint blockWidth = 8u;
    const uint blockHeight = 8u;

    std::vector<char> deltaBuffer;
    ZSTDCodec zstd;
};

} // namespace quasar

#endif // ALPHA_CODEC_H
