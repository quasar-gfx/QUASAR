#ifndef DATA_RECEIVER_TCP_H
#define DATA_RECEIVER_TCP_H

#include <queue>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <string>

#include <Networking/Socket.h>

namespace quasar {

class DataReceiverTCP {
public:
    struct Stats {
        float timeToReceiveMs = -1.0f;
        float bitrateMbps = 0.0f;
    };

    DataReceiverTCP(const std::string& url, bool nonBlocking = false);
    virtual ~DataReceiverTCP();

    void start();
    void close();

protected:
    std::string url;
    Stats stats;
    std::atomic_bool ready = false;

    virtual void onDataReceived(const std::vector<char>& data) = 0;

private:
    SocketTCP socket;
    std::thread dataRecvingThread;

    void recvData();
};

} // namespace quasar

#endif // DATA_RECEIVER_TCP_H
