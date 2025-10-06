#ifndef ZSTD_CODEC_H
#define ZSTD_CODEC_H

#include <vector>

#include <zstd.h>
#include <Codecs/Codec.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class ZSTDCodec : public Codec {
public:
    ZSTDCodec(
            uint32_t compressionLevel = ZSTD_CLEVEL_DEFAULT,
            uint32_t compressionStrategy = ZSTD_dfast,
            uint32_t numWorkers = 0,
            uint32_t chunkSize = BYTES_PER_MEGABYTE)
        : compressionCtx(ZSTD_createCCtx())
        , decompressionCtx(ZSTD_createDCtx())
    {
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_compressionLevel, compressionLevel);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_strategy, compressionStrategy);

        if (numWorkers > 0) {
            ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_nbWorkers, numWorkers);
            ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_jobSize, chunkSize);
        }
    }
    ~ZSTDCodec() override {
        ZSTD_freeCCtx(compressionCtx);
        ZSTD_freeDCtx(decompressionCtx);
    }

    size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) override {
        double startTime = timeutils::getTimeMicros();

        size_t maxCompressedBytes = ZSTD_compressBound(numBytesUncompressed);
        compressedData.resize(maxCompressedBytes);
        auto res = ZSTD_compress2(
            compressionCtx,
            compressedData.data(), maxCompressedBytes,
            uncompressedData, numBytesUncompressed);

        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return res;
    }

    size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) override {
        double startTime = timeutils::getTimeMicros();

        auto res = ZSTD_decompressDCtx(decompressionCtx,
            decompressedData.data(), decompressedData.size(),
            compressedData, numBytesCompressed);

        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return res;
    }

private:
    ZSTD_CCtx* compressionCtx;
    ZSTD_DCtx* decompressionCtx;
};

} // namespace quasar

#endif // ZSTD_CODEC_H
