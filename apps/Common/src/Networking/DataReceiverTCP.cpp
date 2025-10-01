#include <cstring>
#include <algorithm>
#include <Utils/TimeUtils.h>
#include <Networking/DataReceiverTCP.h>

#define MAX_RECV_SIZE 4096

using namespace quasar;

DataReceiverTCP::DataReceiverTCP(const std::string& url, bool nonBlocking)
    : url(url)
{
    if (url.empty()) {
        return;
    }

    running = true;
    socket = std::make_unique<SocketTCP>(nonBlocking);
    dataRecvingThread = std::thread(&DataReceiverTCP::recvData, this);
}

DataReceiverTCP::~DataReceiverTCP() {
    stop();
}

void DataReceiverTCP::stop() {
    if (url.empty()) {
        return;
    }

    running = false;
    if (dataRecvingThread.joinable()) {
        dataRecvingThread.join();
    }
}

void DataReceiverTCP::recvData() {
    // Attempt to connect to the server
    while (true) {
        if (socket->connect(url) < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else {
            running = true;
            break;
        }
    }

    while (running) {
        int received = 0;
        int expectedSize = 0;

        int receiveStartTime = timeutils::getTimeMicros();

        // Read header first to determine the size of the incoming data packet
        while (running && expectedSize == 0) {
            received = socket->recv(&expectedSize, sizeof(expectedSize), 0);
            if (received < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // retry if the socket is non-blocking and recv would block
                }
                break;
            }
            else if (received == sizeof(expectedSize)) {
                break;
            }
        }

        if (expectedSize == 0) {
            continue;
        }
        data.resize(expectedSize);

        // Read the actual data based on the expected size
        int totalReceived = 0;
        while (running && totalReceived < expectedSize) {
            received = socket->recv(data.data() + totalReceived, expectedSize - totalReceived, 0);
            if (received < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // retry if the socket is non-blocking and recv would block
                }
                break;
            }
            else if (received == 0) {
                // Connection closed
                running = false;
                break;
            }

            totalReceived += received;
        }

        if (totalReceived == expectedSize && !data.empty()) {
            stats.receiveTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - receiveStartTime);
            stats.bitrateMbps = ((sizeof(expectedSize) + data.size() * 8) / timeutils::millisToSeconds(stats.receiveTimeMs)) / BYTES_PER_MEGABYTE;

            if (running) {
                onDataReceived(data); // notify about the received data
            }
        }
    }

    socket->close();
}
