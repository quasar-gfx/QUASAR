#ifndef ALPHA_CODEC_H
#define ALPHA_CODEC_H

#include <Codecs/Codec.h>

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
        compressedData.resize(numBytesUncompressed * 2);

        uint8_t prev = src[0];
        uint8_t currentDelta = prev;
        uint8_t runCount = 1;

        size_t writePos = 0;

        for (int i = 1; i < numBytesUncompressed; i++) {
            uint8_t delta = static_cast<uint8_t>(src[i] - prev);

            if (delta == currentDelta && runCount < 255) {
                runCount++;
            }
            else {
                compressedData[writePos++] = static_cast<char>(runCount);
                compressedData[writePos++] = static_cast<char>(currentDelta);
                currentDelta = delta;
                runCount = 1;
            }

            prev = src[i];
        }

        compressedData[writePos++] = static_cast<char>(runCount);
        compressedData[writePos++] = static_cast<char>(currentDelta);

        compressedData.resize(writePos);

        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return writePos;
    }

    size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) {
        double startTime = timeutils::getTimeMicros();

        const uint8_t* src = static_cast<const uint8_t*>(compressedData);
        const size_t targetSize = static_cast<size_t>(width) * static_cast<size_t>(height);

        uint8_t prev = 0;
        size_t writePos = 0;

        // Decode rle then delta
        int i = 0;
        bool firstIter = true;
        while (i + 1 < numBytesCompressed && writePos < targetSize) {
            uint8_t count = src[i++];
            uint8_t delta = src[i++];

            for (uint8_t j = 0; j < count && writePos < targetSize; ++j) {
                if (firstIter) {
                    prev = delta;
                    firstIter = false;
                }
                else {
                    prev = static_cast<uint8_t>(prev + delta);
                }
                decompressedData[writePos++] = static_cast<char>(prev);
            }
        }

        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return writePos;
    }

private:
    uint width, height;
};

} // namespace quasar

#endif // ALPHA_CODEC_H
