#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <iostream>
#include <thread>
#include <cstring>

#include <Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Camera.h>

#include <CameraPose.h>

class PoseReceiver {
public:
    std::string streamerURL;

    SocketUDP socket;

    Camera* camera;
    Pose currPose;

    explicit PoseReceiver(Camera* camera, std::string streamerURL) : camera(camera), streamerURL(streamerURL), socket(true) {
        socket.setRecvSize(sizeof(Pose));
        socket.bind(streamerURL);
    }

    unsigned int receivePose() {
        int bytesReceived = socket.recv(&currPose, sizeof(Pose), 0);
        if (bytesReceived < 0) {
            return -1; // throw std::runtime_error("Failed to receive data");
        }

        camera->setProjectionMatrix(currPose.proj);
        camera->setViewMatrix(currPose.view);

        return currPose.id;
    }
};

#endif // POSE_RECEIVER_H
