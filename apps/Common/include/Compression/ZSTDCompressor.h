#ifndef ZSTD_COMPRESSOR_H
#define ZSTD_COMPRESSOR_H

#include <vector>

#include <zstd.h>

#include <Compression/Compressor.h>

class ZSTDCompressor : public Compressor {
public:
    ZSTDCompressor(uint32_t compressionLevel = 20, uint32_t compressionStrategy = 4, uint32_t numWorkers = 0) {
        compressionCtx = ZSTD_createCCtx();
        decompressionCtx = ZSTD_createDCtx();
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_compressionLevel, compressionLevel);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_strategy, compressionStrategy);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_nbWorkers, numWorkers);
    }
    ~ZSTDCompressor() override = default;

    unsigned int compress(const void* uncompressedData, std::vector<char> &compressedData, unsigned int numBytesUncompressed) override {
        uint32_t maxCompressedBytes = ZSTD_compressBound(numBytesUncompressed);
        compressedData.resize(maxCompressedBytes);

        return ZSTD_compress2(
            compressionCtx,
            compressedData.data(),
            maxCompressedBytes,
            uncompressedData,
            numBytesUncompressed);
    }

    unsigned int decompress(const std::vector<char> &compressedData, std::vector<char> &decompressedData) override {
        return ZSTD_decompressDCtx(decompressionCtx,
            decompressedData.data(),
            decompressedData.size(),
            compressedData.data(),
            compressedData.size());
    }

private:
    ZSTD_CCtx* compressionCtx;
    ZSTD_DCtx* decompressionCtx;
};

#endif // ZSTD_COMPRESSOR_H
