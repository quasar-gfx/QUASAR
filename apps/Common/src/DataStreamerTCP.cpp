#include <Networking/DataStreamerTCP.h>

using namespace quasar;

DataStreamerTCP::DataStreamerTCP(std::string url, int maxDataSize, bool nonBlocking)
    : url(url)
    , maxDataSize(maxDataSize)
    , socket(nonBlocking)
{
    if (url.empty()) {
        return;
    }

    socket.setReuseAddr();
    socket.setSendSize(maxDataSize);

    dataSendingThread = std::thread(&DataStreamerTCP::sendData, this);
}

DataStreamerTCP::~DataStreamerTCP() {
    ready = false;

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
}

int DataStreamerTCP::send(std::vector<char>& data, bool copy) {
    if (!ready) {
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(m);

        if (copy) {
            datas.push(data);
        }
        else {
            datas.push(std::move(data));
        }

        dataReady = true;
    }
    cv.notify_one();

    return data.size();
}

void DataStreamerTCP::sendData() {
    socket.bind(url);
    socket.listen(1);

    while (true) {
        if ((clientSocketID = socket.accept()) < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else {
            ready = true;
            break;
        }
    }

    while (ready) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return dataReady; });

        if (ready) {
            dataReady = false;
        }
        else {
            break;
        }

        std::vector<char> data = std::move(datas.front());
        datas.pop();

        int startSendTime = timeutils::getTimeMicros();

        // Add header
        int dataSize = data.size();
        std::vector<char> header(sizeof(dataSize));
        std::memcpy(header.data(), &dataSize, sizeof(dataSize));

        // Send header
        int totalSent = 0;
        while (totalSent < header.size()) {
            int sent = socket.sendToClient(clientSocketID, header.data() + totalSent, header.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // retry if the socket is non-blocking and send would block
                }
            }
            else {
                totalSent += sent;
            }
        }

        // Send data
        totalSent = 0;
        while (totalSent < data.size()) {
            int sent = socket.sendToClient(clientSocketID, data.data() + totalSent, data.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // retry if the socket is non-blocking and send would block
                }
            }
            else {
                totalSent += sent;
            }
        }

        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startSendTime);
        stats.bitrateMbps = ((sizeof(dataSize) + data.size() * 8) / timeutils::millisToSeconds(stats.timeToSendMs)) / BYTES_PER_MEGABYTE;
    }

    socket.close();
}
