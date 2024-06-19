#ifndef DATA_RECEIVER_H
#define DATA_RECEIVER_H

#include <map>
#include <deque>
#include <queue>
#include <atomic>
#include <thread>

#include <signal.h>

#include <DataPacket.h>
#include <Socket.h>

class DataReceiverUDP {
public:
    std::string url;

    int maxDataSize;

    explicit DataReceiverUDP(std::string url, int maxDataSize, bool nonBlocking = false)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.setRecvSize(sizeof(DataPacket));
        socket.bind(url);

        running = true;
        dataRecvingThread = std::thread(&DataReceiverUDP::recvData, this);
    }
    ~DataReceiverUDP() {
        close();
    }

    void close() {
        running = false;

        if (dataRecvingThread.joinable()) {
            dataRecvingThread.join();
        }

        socket.close();
    }

    uint8_t* recv(bool first = false) {
        if (results.empty()) {
            return nullptr;
        }

        uint8_t* data = nullptr;

        m.lock();

        if (first) {
            data = results.front();
            results.pop_front();
        }
        else {
            data = results.back();
            results.pop_back();

            // clear all previous results
            while (!results.empty()) {
                delete[] results.front();
                results.pop_front();
            }
        }

        m.unlock();

        return data;
    }

private:
    SocketUDP socket;

    std::thread dataRecvingThread;
    std::mutex m;

    std::atomic_bool running = false;

    std::deque<uint8_t *> results;

    std::map<data_id_t, std::map<packet_id_t, DataPacket>> datas;
    std::map<data_id_t, int> dataSizes;

    int recvPacket(DataPacket* packet) {
        return socket.recv(packet, sizeof(DataPacket), 0);
    }

    void recvData() {
        while (running) {
            DataPacket packet{};
            int received = recvPacket(&packet);
            if (received < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // Retry if the socket is non-blocking and recv would block
                }
            }

            datas[packet.dataID][packet.ID] = packet;
            dataSizes[packet.dataID] += packet.size;

            if (dataSizes[packet.dataID] == maxDataSize) {
                uint8_t* data = new uint8_t[maxDataSize];
                int offset = 0;
                for (auto& p : datas[packet.dataID]) {
                    memcpy(data + offset, p.second.data, p.second.size);
                    offset += p.second.size;
                }

                m.lock();
                results.push_back(data);
                m.unlock();

                datas.erase(packet.dataID);
                dataSizes.erase(packet.dataID);
            }
        }
    }
};

class DataReceiverTCP {
public:
    std::string url;

    int maxDataSize;

    explicit DataReceiverTCP(std::string url, int maxDataSize, bool nonBlocking = false)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        dataRecvingThread = std::thread(&DataReceiverTCP::recvData, this);
    }
    ~DataReceiverTCP() {
        close();
    }

    void close() {
        ready = false;

        if (dataRecvingThread.joinable()) {
            dataRecvingThread.join();
        }

        socket.close();
    }

    uint8_t* recv() {
        if (!ready) {
            return nullptr;
        }

        if (frames.empty()) {
            return nullptr;
        }

        uint8_t* data = frames.front();
        frames.pop();

        return data;
    }

private:
    SocketTCP socket;

    std::thread dataRecvingThread;

    std::atomic_bool ready = false;

    std::queue<uint8_t*> frames;

    void recvData() {
        while (!ready) {
            if (socket.connect(url) < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            else {
                ready = true;
                break;
            }
        }

        while (ready) {
            uint8_t* data = new uint8_t[maxDataSize];
            int currSize = 0;
            while (currSize < maxDataSize) {
                int received = socket.recv(data + currSize, maxDataSize - currSize, 0);
                if (received < 0) {
                    if (errno == EWOULDBLOCK || errno == EAGAIN) {
                        continue; // Retry if the socket is non-blocking and recv would block
                    }
                }

                currSize += received;
            }

            frames.push(data);
        }
    }
};

#endif // DATA_RECEIVER_H
