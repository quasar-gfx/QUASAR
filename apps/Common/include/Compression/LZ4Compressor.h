#ifndef LZ4_COMPRESSOR_H
#define LZ4_COMPRESSOR_H

#include <vector>

#include <lz4.h>

#include <Compression/Compressor.h>

class LZ4Compressor : public Compressor {
public:
    LZ4Compressor() = default;
    ~LZ4Compressor() override = default;

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

#endif // LZ4_COMPRESSOR_H
