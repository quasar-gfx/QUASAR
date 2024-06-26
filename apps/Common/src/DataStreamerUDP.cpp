#include <Networking/DataStreamerUDP.h>

void DataStreamerUDP::close() {
    running = false;

    // send dummy packet to unblock thread
    dataReady = true;
    cv.notify_one();

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
}

int DataStreamerUDP::send(const uint8_t* data) {
    int packetID = 0;
    for (int i = 0; i < maxDataSize; i += PACKET_DATA_SIZE_UDP) {
        DataPacketUDP packet{};
        packet.ID = packetID++;
        packet.dataID = dataID;
        packet.size = std::min(PACKET_DATA_SIZE_UDP, maxDataSize - i);
        std::memcpy(packet.data, data + i, PACKET_DATA_SIZE_UDP);

        {
            std::lock_guard<std::mutex> lock(m);
            packets.push(packet);

            dataReady = true;
        }
        cv.notify_one();
    }

    dataID++;

    return maxDataSize;
}

int DataStreamerUDP::sendPacket(DataPacketUDP* packet) {
    return socket.send(packet, sizeof(DataPacketUDP), 0);
}

void DataStreamerUDP::sendData() {
    while (true) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return dataReady; });

        if (running) {
            dataReady = false;
        }
        else {
            break;
        }

        DataPacketUDP packet = packets.front();
        int sent = sendPacket(&packet);
        if (sent < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                continue; // retry if the socket is non-blocking and send would block
            }
        }
        packets.pop();
    }

    socket.close();
}
