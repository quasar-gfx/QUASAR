#ifndef DATA_STREAMER_UDP_H
#define DATA_STREAMER_UDP_H

#include <thread>
#include <atomic>

#include <Networking/Socket.h>
#include <Networking/DataPacketUDP.h>
#include <Utils/TimeUtils.h>
#include <concurrentqueue/concurrentqueue.h>

namespace quasar {

class DataStreamerUDP {
public:
    std::string url;

    int maxDataSize;

    DataStreamerUDP(std::string url, int maxDataSize, bool nonBlocking = false);
    virtual ~DataStreamerUDP();

    void stop();

    int send(const uint8_t* data);

private:
    std::unique_ptr<SocketUDP> socket;

    std::atomic_bool running{false};
    std::thread dataSendingThread;

    packet_id_t dataID = 0;

    moodycamel::ConcurrentQueue<DataPacketUDP> packets;

    int sendPacket(DataPacketUDP* packet);
    void sendData();
};

} // namespace quasar

#endif // DATA_STREAMER_UDP_H
