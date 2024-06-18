#ifndef DATA_STREAMER_H
#define DATA_STREAMER_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <DataPacket.h>
#include <Socket.h>

class DataStreamer {
public:
    std::string url;

    SocketUDP socket;

    packet_id_t dataID = 0;

    int maxDataSize;

    std::queue<DataPacket> packets;

    explicit DataStreamer(std::string url, int maxDataSize, bool nonBlocking = true)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.setAddress(url);

        running = true;
        dataSendingThread = std::thread(&DataStreamer::sendData, this);
    }
    ~DataStreamer() {
        close();
    }

    void close() {
        running = false;

        // send dummy packet to unblock thread
        dataReady = true;
        cv.notify_one();

        if (dataSendingThread.joinable()) {
            dataSendingThread.join();
        }

        socket.close();
    }

    int send(const uint8_t* data) {
        int packetID = 0;
        for (int i = 0; i < maxDataSize; i += PACKET_DATA_SIZE) {
            DataPacket packet{};
            packet.ID = packetID++;
            packet.dataID = dataID;
            packet.size = std::min(PACKET_DATA_SIZE, maxDataSize - i);
            memcpy(packet.data, data + i, PACKET_DATA_SIZE);

            packets.push(packet);

            std::lock_guard<std::mutex> lock(m);
            dataReady = true;
            cv.notify_one();
        }

        dataID++;

        return maxDataSize;
    }

private:
    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool running = false;

    int sendPacket(DataPacket* packet) {
        return socket.send(packet, sizeof(DataPacket), 0);
    }

    void sendData() {
        while (true) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [this] { return dataReady; });

            if (running) {
                dataReady = false;
            }
            else {
                break;
            }

            DataPacket packet = packets.front();
            sendPacket(&packet);
            packets.pop();
        }
    }
};

#endif // DATA_STREAMER_H
