#ifndef CODEC_H
#define CODEC_H

#include <cstdint>
#include <vector>
#include <future>

namespace quasar {

class Codec {
public:
    struct Stats {
        double compressTimeMs = 0.0;
        double decompressTimeMs = 0.0;
    } stats;

    Codec() = default;
    virtual ~Codec() = default;

    virtual size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) = 0;
    virtual size_t decompress(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) = 0;

    virtual std::future<size_t> compressAsync(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) {
        return std::async(std::launch::async, [this, &compressedData, uncompressedData, numBytesUncompressed]() {
            return this->compress(uncompressedData, compressedData, numBytesUncompressed);
        });
    }
    virtual std::future<size_t> decompressAsync(const void* compressedData, std::vector<char>& decompressedData, size_t numBytesCompressed) {
        return std::async(std::launch::async, [this, &decompressedData, compressedData, numBytesCompressed]() {
            return this->decompress(compressedData, decompressedData, numBytesCompressed);
        });
    }
};

} // namespace quasar

#endif // CODEC_H
