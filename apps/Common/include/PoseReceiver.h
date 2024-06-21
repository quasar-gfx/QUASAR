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

    explicit PoseReceiver(Camera* camera, std::string streamerURL)
            : camera(camera)
            , streamerURL(streamerURL)
            , receiver(streamerURL, sizeof(Pose)) { }

    pose_id_t receivePose(bool setProj = true) {
        std::vector<uint8_t> data = receiver.recv();
        if (data.empty()) {
            return -1;
        }

        if (data.size() < sizeof(Pose)) {
            std::cerr << "Error: Received data size is smaller than expected Pose size" << std::endl;
            return -1;
        }

        memcpy(&currPose, data.data(), sizeof(Pose));

        if (setProj) {
            camera->setProjectionMatrix(currPose.proj);
        }
        camera->setViewMatrix(currPose.view);

        return currPose.id;
    }

private:
    DataReceiverUDP receiver;

    Camera* camera;
    Pose currPose;
};

#endif // POSE_RECEIVER_H
