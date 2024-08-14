#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#include <Networking/Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Camera.h>
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
    
    explicit PoseReceiver(VRCamera* vrcamera, std::string streamerURL)
            : vrcamera(vrcamera)
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

#ifdef VR
        if (setProj) {
            vrcamera->left.setProjectionMatrix(currPose.projL);
            vrcamera->right.setProjectionMatrix(currPose.projR);
        }
        vrcamera->left.setViewMatrix(currPose.viewL);
        vrcamera->right.setViewMatrix(currPose.viewR);
        
        return currPose.id;
#else
        if (setProj) {
            camera->setProjectionMatrix(currPose.proj);
        }
        camera->setViewMatrix(currPose.view);

        return currPose.id;
#endif
    }

private:
    DataReceiverUDP receiver;

    Camera* camera;
    VRCamera* vrcamera;
    Pose currPose;
};

#endif // POSE_RECEIVER_H
