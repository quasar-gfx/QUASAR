#ifndef DATA_PACKET_UDP_H
#define DATA_PACKET_UDP_H

#include <cstdint>

namespace quasar {

#define PACKET_DATA_SIZE_UDP 1024

typedef uint32_t packet_id_t;
typedef uint32_t data_id_t;

struct DataPacketUDP {
    packet_id_t ID;
    data_id_t dataID;
    unsigned int size;
    uint8_t data[PACKET_DATA_SIZE_UDP];
};

} // namespace quasar

#endif // DATA_PACKET_UDP_H
