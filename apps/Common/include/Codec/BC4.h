#ifndef BC4_H
#define BC4_H

#include <cstdint>

#define BC4_BLOCK_SIZE 8

struct BC4Block {
    float max;
    float min;
    uint32_t data[6];
};

#endif // BC4_H
