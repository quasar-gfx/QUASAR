#ifndef DATA_RECEIVER_H
#define DATA_RECEIVER_H

#include <map>
#include <deque>
#include <atomic>
#include <thread>

#include <DataPacket.h>
#include <Socket.h>

class DataReceiver {
public:
    std::string url;

    SocketUDP socket;

    unsigned int maxDataSize;

    explicit DataReceiver(std::string url, unsigned int maxDataSize, bool nonBlocking = true)
            : url(url)
            , maxDataSize(maxDataSize)
            , socket(nonBlocking) {
        socket.setRecvSize(sizeof(DataPacket));
        socket.bind(url);

        running = true;
        dataRecvingThread = std::thread(&DataReceiver::recvData, this);
    }
    ~DataReceiver() {
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
    std::thread dataRecvingThread;
    std::mutex m;

    std::atomic_bool running = false;

    std::deque<uint8_t *> results;

    std::map<packet_id_t, std::map<packet_id_t, DataPacket>> datas;
    std::map<packet_id_t, unsigned int> dataSizes;

    int recvPacket(DataPacket* packet) {
        return socket.recv(packet, sizeof(DataPacket), 0);
    }

    void recvData() {
        while (running) {
            DataPacket packet{};
            if (recvPacket(&packet) < 0) {
                continue;
            }

            datas[packet.ID][packet.dataID] = packet;
            dataSizes[packet.ID] += packet.size;

            if (dataSizes[packet.ID] == maxDataSize) {
                uint8_t* data = new uint8_t[maxDataSize];
                unsigned int offset = 0;
                for (auto& p : datas[packet.ID]) {
                    memcpy(data + offset, p.second.data, p.second.size);
                    offset += p.second.size;
                }

                m.lock();
                results.push_back(data);
                m.unlock();

                datas.erase(packet.ID);
                dataSizes.erase(packet.ID);
            }
        }
    }
};

#endif // DATA_RECEIVER_H
