#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

#include <Networking/Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>
#include <Networking/DataReceiverUDP.h>

#include <CameraPose.h>

class PoseReceiver : public DataReceiverUDP {
public:
    std::string streamerURL;

    PoseReceiver(Camera* camera, std::string streamerURL)
            : camera(camera)
            , streamerURL(streamerURL)
            , DataReceiverUDP(streamerURL, sizeof(Pose)) {
    }

    void onDataReceived(const std::vector<uint8_t>& data) override {
        std::lock_guard<std::mutex> lock(m);

        if (data.size() < sizeof(Pose)) {
            std::cerr << "Error: Received data size is smaller than expected Pose size" << std::endl;
            return;
        }

        Pose pose;
        std::memcpy(&pose, data.data(), sizeof(Pose));

        poses.push_back(pose);
    }

    pose_id_t receivePose(bool setProj = true) {
        std::lock_guard<std::mutex> lock(m);

        if (poses.size() == 0) {
            return -1;
        }

        Pose currPose = poses.back();
        poses.clear();

        if (camera->isVR()) {
            auto* vrCamera = static_cast<VRCamera*>(camera);
            if (setProj) {
                vrCamera->setProjectionMatrices({currPose.stereo.projL, currPose.stereo.projR});
            }
            vrCamera->setViewMatrices({currPose.stereo.viewL, currPose.stereo.viewR});
        }
        else {
            auto* perspectiveCamera = static_cast<PerspectiveCamera*>(camera);
            if (setProj) {
                perspectiveCamera->setProjectionMatrix(currPose.mono.proj);
            }
            perspectiveCamera->setViewMatrix(currPose.mono.view);
        }
        return currPose.id;
    }

private:
    Camera* camera;

    std::mutex m;
    std::deque<Pose> poses;
};

#endif // POSE_RECEIVER_H
