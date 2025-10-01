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

namespace quasar {

class DataReceiverUDP {
public:
    std::string url;

    int maxDataSize;

    DataReceiverUDP(std::string url, int maxDataSize, bool nonBlocking = false);
    virtual ~DataReceiverUDP();

    void stop();

protected:
    std::atomic_bool running{false};
    std::thread dataRecvingThread;

    std::map<packet_id_t, std::map<int, DataPacketUDP>> datas;
    std::map<packet_id_t, int> dataSizes;

    virtual void onDataReceived(const std::vector<char>& data) = 0;

private:
    std::unique_ptr<SocketUDP> socket;

    std::vector<char> data;

    int recvPacket(DataPacketUDP* packet);
    void recvData();
};

} // namespace quasar

#endif // DATA_RECEIVER_UDP_H
