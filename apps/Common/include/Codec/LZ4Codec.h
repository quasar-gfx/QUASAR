#ifndef LZ4_CODEC_H
#define LZ4_CODEC_H

#include <vector>

#include <lz4.h>
#include <Codec/Codec.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class LZ4Codec : public Codec {
public:
    LZ4Codec() = default;
    ~LZ4Codec() override = default;

    size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) override {
        double startTime = timeutils::getTimeMicros();

        size_t maxCompressedBytes = LZ4_compressBound(numBytesUncompressed);
        compressedData.resize(maxCompressedBytes);
        auto res = LZ4_compress_default(
            (const char*)uncompressedData,
            compressedData.data(),
            numBytesUncompressed,
            maxCompressedBytes);

        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return res;
    }

    size_t decompress(const std::vector<char>& compressedData, std::vector<char>& decompressedData) override {
        double startTime = timeutils::getTimeMicros();

        auto res = LZ4_decompress_safe(
            (const char*)compressedData.data(),
            decompressedData.data(),
            compressedData.size(),
            decompressedData.size());

        stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return res;
    }
};

} // namespace quasar

#endif // LZ4_CODEC_H
