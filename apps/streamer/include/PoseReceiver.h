#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <iostream>
#include <thread>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <Camera.h>

class PoseReceiver {
public:
    std::string streamerURL;

    int socketId;
    int clientSocketId;
    struct sockaddr_in streamerAddr;
    socklen_t streamerAddrLen;
    Camera* camera;

    explicit PoseReceiver(Camera* camera, std::string streamerURL) : streamerURL(streamerURL) {
        this->streamerURL = streamerURL;
        this->camera = camera;

        size_t pos = streamerURL.find("://");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid streamer URL");
        }

        std::string addressAndPort = streamerURL.substr(pos + 3);
        pos = addressAndPort.find(':');

        std::string ipAddress = addressAndPort.substr(0, pos);
        std::string portStr = addressAndPort.substr(pos + 1);
        int port = std::stoi(portStr);

        socketId = socket(AF_INET, SOCK_DGRAM, 0);
        if (socketId < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        // Set socket to non-blocking
        int flags = fcntl(socketId, F_GETFL, 0);
        if (flags == -1) {
            throw std::runtime_error("Error getting socket flags");
        }
        if (fcntl(socketId, F_SETFL, flags | O_NONBLOCK) == -1) {
            throw std::runtime_error("Error setting socket to non-blocking");
        }

        int recvBufferSize = 256;
        if (setsockopt(socketId, SOL_SOCKET, SO_RCVBUF, &recvBufferSize, sizeof(recvBufferSize)) < 0) {
            throw std::runtime_error("Failed to set socket receive buffer size");
        }

        streamerAddrLen = sizeof(streamerAddr);
        streamerAddr.sin_family = AF_INET;
        streamerAddr.sin_addr.s_addr = inet_addr(ipAddress.c_str());
        streamerAddr.sin_port = htons(port);

        if (bind(socketId, (struct sockaddr*)&streamerAddr, streamerAddrLen) < 0) {
            throw std::runtime_error("Failed to bind socket");
        }
    }

    void receivePose() {
        glm::mat4 viewMatrix;

        int bytesReceived = recvfrom(socketId, &viewMatrix, sizeof(glm::mat4), MSG_WAITALL, (struct sockaddr*)&streamerAddr, &streamerAddrLen);
        if (bytesReceived < 0) {
            return; // throw std::runtime_error("Failed to receive data");
        }

        // std::cout << glm::to_string(viewMatrix) << std::endl;
        camera->setViewMatrix(viewMatrix);
    }
};

#endif // POSE_RECEIVER_H
