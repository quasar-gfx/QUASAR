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

    explicit DataReceiverUDP(std::string url, int maxDataSize, bool nonBlocking = false)
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

    std::vector<uint8_t> recv(bool first = false);

private:
    SocketUDP socket;

    std::thread dataRecvingThread;
    std::mutex m;

    std::atomic_bool running = false;

    std::deque<std::vector<uint8_t>> results;

    std::map<packet_id_t, std::map<int, DataPacketUDP>> datas;
    std::map<packet_id_t, int> dataSizes;

    int recvPacket(DataPacketUDP* packet);
    void recvData();
};

#endif // DATA_RECEIVER_UDP_H
