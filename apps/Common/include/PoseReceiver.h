#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <chrono>
#include <thread>
#include <cstring>

#include <spdlog/spdlog.h>

#include <Networking/Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>
#include <Networking/DataReceiverUDP.h>

#include <CameraPose.h>

namespace quasar {

class PoseReceiver : public DataReceiverUDP {
public:
    std::string streamerURL;

    PoseReceiver(Camera* camera, std::string streamerURL, float poseDropThresMs = 50.0f)
            : camera(camera)
            , streamerURL(streamerURL)
            , poseDropThresUs(timeutils::millisToMicros(poseDropThresMs))
            , DataReceiverUDP(streamerURL, sizeof(Pose)) {
        spdlog::info("Created PoseReceiver that recvs from URL: {}", streamerURL);
    }

    void onDataReceived(const std::vector<uint8_t>& data) override {
        std::lock_guard<std::mutex> lock(m);

        if (data.size() < sizeof(Pose)) {
            spdlog::warn("Received data size is smaller than expected Pose size");
            return;
        }

        Pose newPose;
        std::memcpy(&newPose, data.data(), sizeof(Pose));

        if (newPose.timestamp - currPose.timestamp > poseDropThresUs) {
            currPose = newPose;
            currPoseDirty = true;
        }
    }

    pose_id_t receivePose(bool setProj = true) {
        std::lock_guard<std::mutex> lock(m);

        if (!currPoseDirty) {
            return -1;
        }
        currPoseDirty = false;

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
    float poseDropThresUs;

    std::mutex m;
    bool currPoseDirty = false;
    Pose currPose;
};

} // namespace quasar

#endif // POSE_RECEIVER_H
