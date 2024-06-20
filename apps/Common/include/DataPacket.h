#ifndef PACKET_H
#define PACKET_H

#include <cstdint>

#define PACKET_DATA_SIZE 1024

typedef uint32_t packet_id_t;
typedef uint32_t data_id_t;

struct DataPacket {
    packet_id_t ID;
    data_id_t dataID;
    unsigned int size;
    uint8_t data[PACKET_DATA_SIZE];
};

#endif // PACKET_H
