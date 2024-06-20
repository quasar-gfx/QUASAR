#ifndef DATA_STREAMER_UDP_H
#define DATA_STREAMER_UDP_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <DataPacket.h>
#include <Socket.h>

class DataStreamerUDP {
public:
    std::string url;

    int maxDataSize;

    explicit DataStreamerUDP(std::string url, int maxDataSize, bool nonBlocking = false)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.setAddress(url);

        running = true;
        dataSendingThread = std::thread(&DataStreamerUDP::sendData, this);
    }
    ~DataStreamerUDP() {
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
    SocketUDP socket;

    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool running = false;

    packet_id_t dataID = 0;

    std::queue<DataPacket> packets;

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
            int sent = sendPacket(&packet);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // Retry if the socket is non-blocking and send would block
                }
            }
            packets.pop();
        }
    }
};

class DataStreamerTCP {
public:
    std::string url;

    int maxDataSize;

    explicit DataStreamerTCP(std::string url, int maxDataSize, bool nonBlocking = false, int sendSize = 65535)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.bind(url);
        socket.setReuseAddrPort();
        socket.setSendSize(sendSize);
        socket.listen(1);

        dataSendingThread = std::thread(&DataStreamerTCP::sendData, this);
    }
    ~DataStreamerTCP() {
        close();
    }

    void close() {
        ready = false;

        if (dataSendingThread.joinable()) {
            dataSendingThread.join();
        }

        socket.close();
    }

    int send(const uint8_t* data, bool copy = false) {
        if (!ready) {
            return -1;
        }

        if (copy) {
            uint8_t* dataCopy = new uint8_t[maxDataSize];
            memcpy(dataCopy, data, maxDataSize);
            datas.push(dataCopy);
        }
        else {
            datas.push(data);
        }

        return maxDataSize;
    }

private:
    SocketTCP socket;

    std::thread dataSendingThread;

    std::atomic_bool ready = false;

    std::queue<const uint8_t*> datas;

    void sendData() {
        while (true) {
            if (socket.accept() < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            else {
                ready = true;
                break;
            }
        }

        while (ready) {
            if (datas.empty()) {
                continue;
            }

            const uint8_t* data = datas.front();
            datas.pop();

            int sent = socket.send(data, maxDataSize, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // Retry if the socket is non-blocking and send would block
                }
            }
        }
    }
};

#endif // DATA_STREAMER_UDP_H
