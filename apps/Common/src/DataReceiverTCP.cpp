#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>
#include <cstring>
#include <algorithm>
#include <chrono>

#define MAX_RECV_SIZE 4096

using namespace quasar;

DataReceiverTCP::DataReceiverTCP(const std::string& url, bool nonBlocking)
    : url(url)
    , socket(nonBlocking)
{
    start();
}

DataReceiverTCP::~DataReceiverTCP() {
    close();
}

void DataReceiverTCP::start() {
    ready = true;
    dataRecvingThread = std::thread(&DataReceiverTCP::recvData, this);
}

void DataReceiverTCP::close() {
    ready = false;
    if (dataRecvingThread.joinable()) {
        dataRecvingThread.join();
    }
    socket.close();
}

void DataReceiverTCP::recvData() {
    // Attempt to connect to the server
    while (true) {
        if (socket.connect(url) < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else {
            ready = true;
            break;
        }
    }

    while (ready) {
        std::vector<char> data;
        char buffer[MAX_RECV_SIZE];

        int received = 0;
        int expectedSize = 0;

        int receiveStartTime = timeutils::getTimeMicros();

        // Read header first to determine the size of the incoming data packet
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

        // Read the actual data based on the expected size
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
                // Connection closed
                ready = false;
                break;
            }
        }

        if (totalReceived == expectedSize && !data.empty()) {
            stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - receiveStartTime);
            stats.bitrateMbps = ((sizeof(expectedSize) + data.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / BYTES_PER_MEGABYTE;

            onDataReceived(std::move(data)); // notify about the received data
        }
    }

    socket.close();
}
