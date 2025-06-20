#ifndef DATA_STREAMER_TCP_H
#define DATA_STREAMER_TCP_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <Utils/TimeUtils.h>

#include <Networking/DataPacketUDP.h>
#include <Networking/Socket.h>

namespace quasar {

class DataStreamerTCP {
public:
    std::string url;

    int maxDataSize;

    struct Stats {
        double timeToSendMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DataStreamerTCP(std::string url, int maxDataSize = 65535, bool nonBlocking = false)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        if (url.empty()) {
            return;
        }

        socket.setReuseAddr();
        socket.setSendSize(maxDataSize);

        dataSendingThread = std::thread(&DataStreamerTCP::sendData, this);
    }

    ~DataStreamerTCP() {
        ready = false;

        if (dataSendingThread.joinable()) {
            dataSendingThread.join();
        }
    }

    int send(std::vector<char>& data, bool copy = false);

private:
    SocketTCP socket;

    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool ready = false;

    std::queue<std::vector<char>> datas;

    void sendData();

    int clientSocketID = -1;
};

} // namespace quasar

#endif // DATA_STREAMER_TCP_H
