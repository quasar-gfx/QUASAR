#include <Networking/DataReceiverUDP.h>

void DataReceiverUDP::close() {
    running = false;

    if (dataRecvingThread.joinable()) {
        dataRecvingThread.join();
    }
}

std::vector<uint8_t> DataReceiverUDP::recv(bool first) {
    std::lock_guard<std::mutex> lock(m);

    if (results.empty()) {
        return {};
    }

    std::vector<uint8_t> data;

    if (first) {
        data = std::move(results.front());
        results.pop_front();
    }
    else {
        data = std::move(results.back());
        results.pop_back();

        // Clear all previous results
        while (!results.empty()) {
            results.pop_front();
        }
    }

    return data;
}

int DataReceiverUDP::recvPacket(DataPacketUDP* packet) {
    return socket.recv(packet, sizeof(DataPacketUDP), 0);
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
            std::vector<uint8_t> data(maxDataSize);
            int offset = 0;
            for (auto& p : datas[packet.dataID]) {
                memcpy(data.data() + offset, p.second.data, p.second.size);
                offset += p.second.size;
            }

            {
                std::unique_lock<std::mutex> lock(m);
                results.push_back(std::move(data));
            }

            datas.erase(packet.dataID);
            dataSizes.erase(packet.dataID);
        }
    }

    socket.close();
}
