#include <Networking/DataStreamerTCP.h>

void DataStreamerTCP::close() {
    ready = false;

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
}

int DataStreamerTCP::send(std::vector<uint8_t> data, bool copy) {
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
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return dataReady; });

        if (ready) {
            dataReady = false;
        }
        else {
            break;
        }

        std::vector<uint8_t> data = std::move(datas.front());
        datas.pop();

        int startSendTime = timeutils::getCurrTimeMs();

        // add header
        int dataSize = data.size();
        std::vector<uint8_t> header(sizeof(dataSize));
        memcpy(header.data(), &dataSize, sizeof(dataSize));

        // send header
        int totalSent = 0;
        while (totalSent < header.size()) {
            int sent = socket.send(header.data() + totalSent, header.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // retry if the socket is non-blocking and send would block
                }
            }
            else {
                totalSent += sent;
            }
        }

        // send data
        totalSent = 0;
        while (totalSent < data.size()) {
            int sent = socket.send(data.data() + totalSent, data.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) {
                    continue; // retry if the socket is non-blocking and send would block
                }
            }
            else {
                totalSent += sent;
            }
        }

        stats.timeToSendMs = (timeutils::getCurrTimeMs() - startSendTime);
        stats.bitrateMbps = ((data.size() * 8) / (stats.timeToSendMs * MILLISECONDS_IN_SECOND));
    }

    socket.close();
}
