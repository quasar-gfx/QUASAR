#ifndef DATA_STREAMER_TCP_H
#define DATA_STREAMER_TCP_H

#include <vector>
#include <thread>
#include <atomic>

#include <Networking/Socket.h>
#include <Networking/DataPacketUDP.h>
#include <Utils/TimeUtils.h>
#include <concurrentqueue/concurrentqueue.h>

namespace quasar {

class DataStreamerTCP {
public:
    std::string url;

    int maxDataSize;

    struct Stats {
        double sendTimeMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DataStreamerTCP(std::string url, int maxDataSize = 65535, bool nonBlocking = false);
    virtual ~DataStreamerTCP();

    int send(std::vector<char>& data, bool copy = false);
    void stop();

private:
    std::unique_ptr<SocketTCP> socket;

    std::thread dataSendingThread;

    std::atomic_bool ready = false;
    std::atomic_bool shouldTerminate = false;

    moodycamel::ConcurrentQueue<std::vector<char>> datas;

    void sendData();

    int clientSocketID = -1;
};

} // namespace quasar

#endif // DATA_STREAMER_TCP_H
