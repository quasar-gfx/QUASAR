#ifndef DATA_RECEIVER_TCP_H
#define DATA_RECEIVER_TCP_H

#include <map>
#include <deque>
#include <queue>
#include <atomic>
#include <thread>

#include <Networking/Socket.h>

class DataReceiverTCP {
public:
    std::string url;

    struct Stats {
        float timeToReceiveMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    DataReceiverTCP(std::string url, bool nonBlocking = false)
            : url(url)
            , socket(nonBlocking) {
        dataRecvingThread = std::thread(&DataReceiverTCP::recvData, this);
    }

    ~DataReceiverTCP() {
        close();
    }

    void close();

    std::vector<uint8_t> recv();

private:
    SocketTCP socket;

    std::thread dataRecvingThread;

    std::atomic_bool ready = false;

    std::queue<std::vector<uint8_t>> frames;

    void recvData();
};

#endif // DATA_RECEIVER_TCP_H
