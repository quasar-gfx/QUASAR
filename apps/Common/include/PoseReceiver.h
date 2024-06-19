#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#include <Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Camera.h>
#include <DataReceiver.h>

#include <CameraPose.h>

class PoseReceiver {
public:
    std::string streamerURL;

    DataReceiverUDP receiver;

    Camera* camera;
    Pose currPose;

    explicit PoseReceiver(Camera* camera, std::string streamerURL)
            : camera(camera)
            , streamerURL(streamerURL)
            , receiver(streamerURL, sizeof(Pose)) { }

    pose_id_t receivePose(bool setProj = true) {
        uint8_t* data = receiver.recv();
        if (data == nullptr) {
            return -1;
        }

        memcpy(&currPose, data, sizeof(Pose));
        delete[] data;

        if (setProj) {
            camera->setProjectionMatrix(currPose.proj);
        }
        camera->setViewMatrix(currPose.view);

        return currPose.id;
    }
};

#endif // POSE_RECEIVER_H
