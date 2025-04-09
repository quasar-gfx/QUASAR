#ifndef LZ4_CODEC_H
#define LZ4_CODEC_H

#include <vector>

#include <lz4.h>
#include <Codec/Codec.h>

namespace quasar {

class LZ4Codec : public Codec {
public:
    LZ4Codec() = default;
    ~LZ4Codec() override = default;

    unsigned int compress(const void* uncompressedData, std::vector<char> &compressedData, unsigned int numBytesUncompressed) override {
        size_t maxCompressedBytes = LZ4_compressBound(numBytesUncompressed);
        compressedData.resize(maxCompressedBytes);
        return LZ4_compress_default(
            (const char*)uncompressedData,
            compressedData.data(),
            numBytesUncompressed,
            maxCompressedBytes);
    }

    unsigned int decompress(const std::vector<char> &compressedData, std::vector<char> &decompressedData) override {
        return LZ4_decompress_safe(
            (const char*)compressedData.data(),
            decompressedData.data(),
            compressedData.size(),
            decompressedData.size());
    }
};

} // namespace quasar

#endif // LZ4_CODEC_H
