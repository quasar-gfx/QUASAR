#ifndef POSE_STREAMER_H
#define POSE_STREAMER_H

#include <iostream>
#include <thread>
#include <cstring>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/epsilon.hpp>

#include <Camera.h>

class PoseStreamer {
public:
    std::string receiverURL;

    int socketId;
    struct sockaddr_in recieverAddr;
    socklen_t recieverAddrLen;
    hostent* server;
    Camera* camera;
    glm::mat4 prevViewMatrix;

    explicit PoseStreamer(Camera* camera, std::string receiverURL) : receiverURL(receiverURL) {
        this->camera = camera;

        size_t pos = receiverURL.find("://");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid streamer URL");
        }

        std::string addressAndPort = receiverURL.substr(pos + 3);
        pos = addressAndPort.find(':');

        std::string ipAddress = addressAndPort.substr(0, pos);
        std::string portStr = addressAndPort.substr(pos + 1);
        int port = std::stoi(portStr);

        socketId = socket(AF_INET, SOCK_DGRAM, 0);
        if (socketId < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        recieverAddrLen = sizeof(recieverAddr);
        recieverAddr.sin_family = AF_INET;
        recieverAddr.sin_addr.s_addr = inet_addr(ipAddress.c_str());
        recieverAddr.sin_port = htons(port);
    }

    bool epsilonEqual(const glm::mat4& mat1, const glm::mat4& mat2, float epsilon = 0.001f) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(mat1[i][j] - mat2[i][j]) > epsilon)
                    return false;
            }
        }
        return true;
    }

    void sendPose() {
        glm::mat4 viewMatrix = camera->getViewMatrix();
        // std::cout << glm::to_string(viewMatrix) << std::endl;

        if (epsilonEqual(viewMatrix, prevViewMatrix)) {
            return;
        }

        int bytesSent = sendto(socketId, &viewMatrix, sizeof(glm::mat4), MSG_WAITALL, (struct sockaddr*)&recieverAddr, recieverAddrLen);
        if (bytesSent < 0) {
            std::cout << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to send data");
        }

        prevViewMatrix = viewMatrix;
    }
};

#endif // POSE_STREAMER_H
