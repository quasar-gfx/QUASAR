#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#include <Networking/Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <PerspectiveCamera.h>
#include <VRCamera.h>
#include <Networking/DataReceiverUDP.h>

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

        std::memcpy(&currPose, data.data(), sizeof(Pose));

        if (VRCamera* vrCamera = dynamic_cast<VRCamera*>(camera)) {
            if (setProj) {
                vrCamera->setProjectionMatrix(currPose.vr.projL);
            }
            vrCamera->left->setViewMatrix(currPose.vr.viewL);
            vrCamera->right->setViewMatrix(currPose.vr.viewR);
        } else if (PerspectiveCamera* perspectiveCamera = dynamic_cast<PerspectiveCamera*>(camera)) {
            if (setProj) {
                perspectiveCamera->setProjectionMatrix(currPose.non_vr.proj);
            }
            perspectiveCamera->setViewMatrix(currPose.non_vr.view);
        } else {
            std::cerr << "Error: camera is neither a VRCamera nor a PerspectiveCamera instance" << std::endl;
        }
        return currPose.id;
    }

private:
    DataReceiverUDP receiver;

    Camera* camera;
    Pose currPose;
};

#endif // POSE_RECEIVER_H
