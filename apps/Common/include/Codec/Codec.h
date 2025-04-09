#ifndef CODEC_H
#define CODEC_H

#include <cstdint>
#include <vector>

namespace quasar {

class Codec {
public:
    Codec() = default;
    virtual ~Codec() = default;

    virtual unsigned int compress(const void* uncompressedData, std::vector<char> &compressedData, unsigned int numBytesUncompressed) = 0;
    virtual unsigned int decompress(const std::vector<char> &compressedData, std::vector<char> &decompressedData) = 0;
};

} // namespace quasar

#endif // CODEC_H
