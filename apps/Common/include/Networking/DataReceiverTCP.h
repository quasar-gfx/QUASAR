#ifndef DATA_RECEIVER_TCP_H
#define DATA_RECEIVER_TCP_H

#include <vector>
#include <atomic>
#include <thread>
#include <string>
#include <functional>

#include <Networking/Socket.h>

namespace quasar {

class DataReceiverTCP {
public:
    struct Stats {
        double receiveTimeMs = 0.0;
        double bitrateMbps = 0.0;
    };

    DataReceiverTCP(const std::string& url, bool nonBlocking = false);
    virtual ~DataReceiverTCP();

    void stop();

protected:
    std::string url;
    Stats stats;
    std::atomic_bool running = false;
    std::thread dataRecvingThread;

    virtual void onDataReceived(const std::vector<char>& data) = 0;

private:
    std::unique_ptr<SocketTCP> socket;

    std::vector<char> data;

    void recvData();
};

} // namespace quasar

#endif // DATA_RECEIVER_TCP_H
