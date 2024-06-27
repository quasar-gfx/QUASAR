#include <Networking/DataReceiverTCP.h>

#include <Utils/TimeUtils.h>

#define MAX_RECV_SIZE 4096

void DataReceiverTCP::close() {
    ready = false;

    if (dataRecvingThread.joinable()) {
        dataRecvingThread.join();
    }
}

std::vector<uint8_t> DataReceiverTCP::recv() {
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

void DataReceiverTCP::recvData() {
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
        uint8_t buffer[MAX_RECV_SIZE];

        int received = 0;
        int expectedSize = 0;

        int receiveStartTime = timeutils::getCurrTimeMicros();

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
                std::memcpy(&expectedSize, buffer, sizeof(expectedSize));
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
            received = socket.recv(buffer, std::min(MAX_RECV_SIZE, expectedSize - totalReceived), 0);
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
            stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getCurrTimeMicros() - receiveStartTime);
            stats.bitrateMbps = ((data.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / MBPS_TO_BPS;
            frames.push(std::move(data));
        }
    }

    socket.close();
}
