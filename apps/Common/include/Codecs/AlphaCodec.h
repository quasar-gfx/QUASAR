#ifndef ALPHA_CODEC_H
#define ALPHA_CODEC_H

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
    {}
    ~AlphaCodec() = default;

    size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) {
        double startTime = timeutils::getTimeMicros();

        // Delta-encode the input
        const uint8_t* src = static_cast<const uint8_t*>(uncompressedData);

        deltaBuffer.resize(numBytesUncompressed);

        if (numBytesUncompressed > 0) {
            deltaBuffer[0] = static_cast<char>(src[0]);
            for (size_t i = 1; i < numBytesUncompressed; ++i) {
                uint8_t delta = static_cast<uint8_t>(src[i] - src[i - 1]);
                deltaBuffer[i] = static_cast<char>(delta);
            }
        }

        size_t written = zstd.compress(deltaBuffer.data(), compressedData, deltaBuffer.size());
        compressedData.resize(written);

        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return written;
    }

    size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) {
        double startTime = timeutils::getTimeMicros();

        const size_t targetSize = static_cast<size_t>(width) * static_cast<size_t>(height);

        // Decompress with ZSTD
        deltaBuffer.resize(targetSize);

        size_t decompressedBytes = zstd.decompress(compressedData, deltaBuffer, numBytesCompressed);

        size_t writePos = 0;
        uint8_t prev = 0;
        if (decompressedBytes > 0 && targetSize > 0) {
            // First byte is the initial value
            prev = static_cast<uint8_t>(deltaBuffer[0]);
            decompressedData[writePos++] = static_cast<char>(prev);

            for (size_t i = 1; i < std::min(decompressedBytes, targetSize); ++i) {
                uint8_t delta = static_cast<uint8_t>(deltaBuffer[i]);
                prev = static_cast<uint8_t>(prev + delta);
                decompressedData[writePos++] = static_cast<char>(prev);
            }
        }

        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return writePos;
    }

private:
    uint width, height;

    std::vector<char> deltaBuffer;
    ZSTDCodec zstd;
};

} // namespace quasar

#endif // ALPHA_CODEC_H
