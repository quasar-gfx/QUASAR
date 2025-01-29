#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <cstdint>
#include <vector>

class Compressor {
public:
    Compressor() = default;
    virtual ~Compressor() = default;

    virtual unsigned int compress(const void* uncompressedData, std::vector<char> &compressedData, unsigned int numBytesUncompressed) = 0;
    virtual unsigned int decompress(const std::vector<char> &compressedData, std::vector<char> &decompressedData) = 0;
};

#endif // COMPRESSOR_H
