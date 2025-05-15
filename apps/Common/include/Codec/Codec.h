#ifndef CODEC_H
#define CODEC_H

#include <cstdint>
#include <vector>

namespace quasar {

class Codec {
public:
    Codec() = default;
    virtual ~Codec() = default;

    virtual uint compress(const void* uncompressedData, std::vector<char>& compressedData, uint numBytesUncompressed) = 0;
    virtual uint decompress(const std::vector<char>& compressedData, std::vector<char>& decompressedData) = 0;
};

} // namespace quasar

#endif // CODEC_H
