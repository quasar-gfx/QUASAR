#include <Networking/DataReceiverUDP.h>

using namespace quasar;

DataReceiverUDP::DataReceiverUDP(std::string url, int maxDataSize, bool nonBlocking)
    : url(url)
    , maxDataSize(maxDataSize)
{
    if (url.empty()) {
        return;
    }

    socket = std::make_unique<SocketUDP>(nonBlocking);
    socket->bind(url);

    running = true;
    dataRecvingThread = std::thread(&DataReceiverUDP::recvData, this);
}

DataReceiverUDP::~DataReceiverUDP() {
    stop();
}

void DataReceiverUDP::stop() {
    if (url.empty()) {
        return;
    }

    running = false;
    if (dataRecvingThread.joinable()) {
        dataRecvingThread.join();
    }
}

int DataReceiverUDP::recvPacket(DataPacketUDP* packet) {
    return socket->recv(packet, sizeof(DataPacketUDP), 0);
}

void DataReceiverUDP::recvData() {
    while (running) {
        DataPacketUDP packet{};
        int received = recvPacket(&packet);
        if (received < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                continue; // retry if the socket is non-blocking and recv would block
            }
        }

        datas[packet.dataID][packet.ID] = packet;
        dataSizes[packet.dataID] += packet.size;

        if (dataSizes[packet.dataID] == maxDataSize) {
            data.resize(maxDataSize);

            int offset = 0;
            for (auto& p : datas[packet.dataID]) {
                std::memcpy(data.data() + offset, p.second.data, p.second.size);
                offset += p.second.size;
            }

            if (running) {
                onDataReceived(data); // notify about the received data
            }

            datas.erase(packet.dataID);
            dataSizes.erase(packet.dataID);
        }
    }

    socket->close();
}
