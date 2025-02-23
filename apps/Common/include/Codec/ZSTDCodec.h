#ifndef ZSTD_CODEC_H
#define ZSTD_CODEC_H

#include <vector>

#include <zstd.h>

#include <Codec/Codec.h>

class ZSTDCodec : public Codec {
public:
    ZSTDCodec(
            uint32_t compressionLevel = 3,
            uint32_t compressionStrategy = ZSTD_greedy,
            uint32_t numWorkers = 0) {
        compressionCtx = ZSTD_createCCtx();
        decompressionCtx = ZSTD_createDCtx();

        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_compressionLevel, compressionLevel);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_strategy, compressionStrategy);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_nbWorkers, numWorkers);

        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_enableLongDistanceMatching, 0);
    }
    ~ZSTDCodec() override = default;

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

#endif // ZSTD_CODEC_H
