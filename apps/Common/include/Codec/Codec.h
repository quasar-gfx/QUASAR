#ifndef CODEC_H
#define CODEC_H

#include <cstdint>
#include <vector>
#include <future>

namespace quasar {

class Codec {
public:
    struct Stats {
        double timeToCompressMs = 0.0;
        double timeToDecompressMs = 0.0;
    } stats;

    Codec() = default;
    virtual ~Codec() = default;

    virtual size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) = 0;
    virtual size_t decompress(const std::vector<char>& compressedData, std::vector<char>& decompressedData) = 0;

    virtual std::future<size_t> compressAsync(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) {
        return std::async(std::launch::async, [=, &compressedData, this]() {
            return this->compress(uncompressedData, compressedData, numBytesUncompressed);
        });
    }
    virtual std::future<size_t> decompressAsync(const std::vector<char>& compressedData, std::vector<char>& decompressedData) {
        return std::async(std::launch::async, [=, &decompressedData, this]() {
            return this->decompress(compressedData, decompressedData);
        });
    }
};

} // namespace quasar

#endif // CODEC_H
