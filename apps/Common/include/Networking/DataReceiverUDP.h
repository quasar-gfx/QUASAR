#ifndef DATA_RECEIVER_UDP_H
#define DATA_RECEIVER_UDP_H

#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>

#include <Utils/TimeUtils.h>

#include <Networking/DataPacketUDP.h>
#include <Networking/Socket.h>

class DataReceiverUDP {
public:
    std::string url;

    int maxDataSize;

    DataReceiverUDP(std::string url, int maxDataSize, bool nonBlocking = false)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.bind(url);

        running = true;
        dataRecvingThread = std::thread(&DataReceiverUDP::recvData, this);
    }
    ~DataReceiverUDP() {
        close();
    }

    void close();

protected:
    std::thread dataRecvingThread;

    std::atomic_bool running = false;

    std::map<packet_id_t, std::map<int, DataPacketUDP>> datas;
    std::map<packet_id_t, int> dataSizes;

    virtual void onDataReceived(const std::vector<uint8_t>& data) = 0;

private:
    SocketUDP socket;

    int recvPacket(DataPacketUDP* packet);
    void recvData();
};

#endif // DATA_RECEIVER_UDP_H
