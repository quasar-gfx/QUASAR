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
    }

    std::vector<uint8_t> recv(bool first = false) {
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

private:
    SocketUDP socket;

    std::thread dataRecvingThread;
    std::mutex m;

    std::atomic_bool running = false;

    std::deque<std::vector<uint8_t>> results;

    std::map<packet_id_t, std::map<int, DataPacket>> datas;
    std::map<packet_id_t, int> dataSizes;

    int recvPacket(DataPacket* packet) {
        return socket.recv(packet, sizeof(DataPacket), 0);
    }

    void recvData() {
        while (running) {
            DataPacket packet{};
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

                std::lock_guard<std::mutex> lock(m);
                results.push_back(std::move(data));

                datas.erase(packet.dataID);
                dataSizes.erase(packet.dataID);
            }
        }

        socket.close();
    }
};


class DataReceiverTCP {
public:
    std::string url;

    explicit DataReceiverTCP(std::string url, bool nonBlocking = false)
            : url(url)
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
    }

    std::vector<uint8_t> recv() {
        if (!ready) {
            return {};
        }

        if (frames.empty()) {
            return {};
        }

        std::vector<uint8_t> data = std::move(frames.front());
        frames.pop();

        return data;
    }

private:
    SocketTCP socket;

    std::thread dataRecvingThread;

    std::atomic_bool ready = false;

    std::queue<std::vector<uint8_t>> frames;

    void recvData() {
        while (true) {
            if (socket.connect(url) < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                ready = true;
                break;
            }
        }

        while (ready) {
            std::vector<uint8_t> data;
            uint8_t buffer[PACKET_DATA_SIZE];

            int received = 0;
            int expectedSize = 0;

            // read header first (includes size of the data packet)
            while (ready && expectedSize == 0) {
                received = socket.recv(buffer, sizeof(expectedSize), 0);
                if (received < 0) {
                    if (errno == EWOULDBLOCK || errno == EAGAIN) {
                        continue; // retry if the socket is non-blocking and recv would block
                    }
                    break;
                }

                if (received == sizeof(expectedSize)) {
                    memcpy(&expectedSize, buffer, sizeof(expectedSize));
                    data.reserve(expectedSize);
                    break;
                }
            }

            if (expectedSize == 0) {
                continue;
            }

            // Now read the actual data based on the expected size
            int totalReceived = 0;
            while (ready && totalReceived < expectedSize) {
                received = socket.recv(buffer, std::min(PACKET_DATA_SIZE, expectedSize - totalReceived), 0);
                if (received < 0) {
                    if (errno == EWOULDBLOCK || errno == EAGAIN) {
                        continue; // retry if the socket is non-blocking and recv would block
                    }
                    break;
                }

                data.insert(data.end(), buffer, buffer + received);
                totalReceived += received;

                if (received == 0) {
                    // connection closed
                    ready = false;
                    break;
                }
            }

            if (totalReceived == expectedSize && !data.empty()) {
                frames.push(std::move(data));
            }
        }

        socket.close();
    }
};

#endif // DATA_RECEIVER_H
