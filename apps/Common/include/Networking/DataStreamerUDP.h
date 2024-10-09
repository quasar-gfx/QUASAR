#ifndef DATA_STREAMER_UDP_H
#define DATA_STREAMER_UDP_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <Utils/TimeUtils.h>

#include <Networking/DataPacketUDP.h>
#include <Networking/Socket.h>

class DataStreamerUDP {
public:
    std::string url;

    int maxDataSize;

    DataStreamerUDP(std::string url, int maxDataSize, bool nonBlocking = false)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.setAddress(url);

        running = true;
        dataSendingThread = std::thread(&DataStreamerUDP::sendData, this);
    }
    ~DataStreamerUDP() {
        close();
    }

    void close();

    int send(const uint8_t* data);

private:
    SocketUDP socket;

    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool running{false};

    packet_id_t dataID = 0;

    std::queue<DataPacketUDP> packets;

    int sendPacket(DataPacketUDP* packet);
    void sendData();
};

#endif // DATA_STREAMER_UDP_H
