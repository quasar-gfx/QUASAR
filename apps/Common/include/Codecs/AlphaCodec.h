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

        compressedData.resize(numBytesUncompressed);
        std::memcpy(compressedData.data(), uncompressedData, numBytesUncompressed);

        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return numBytesUncompressed;
    }

    size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) {
        double startTime = timeutils::getTimeMicros();

        decompressedData.resize(numBytesCompressed);
        std::memcpy(decompressedData.data(), compressedData, numBytesCompressed);

        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return numBytesCompressed;
    }

private:
    uint width, height;
};

} // namespace quasar

#endif // ALPHA_CODEC_H
