#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <chrono>
#include <thread>
#include <cstring>
#include <deque>

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

    PoseReceiver(Camera* camera, const std::string& streamerURL, float poseDropThresMs = 50.0f, size_t maxSavedPoses = 2)
        : camera(camera)
        , streamerURL(streamerURL)
        , poseDropThresUs(timeutils::millisToMicros(poseDropThresMs))
        , maxSavedPoses(maxSavedPoses)
        , DataReceiverUDP(streamerURL, sizeof(Pose))
    {
        spdlog::info("Created PoseReceiver that recvs from URL: {}", this->streamerURL);
    }

    void onDataReceived(const std::vector<uint8_t>& data) override {
        std::lock_guard<std::mutex> lock(m);

        if (data.size() < sizeof(Pose)) {
            spdlog::warn("Received data size is smaller than expected Pose size");
            return;
        }

        Pose newPose;
        std::memcpy(&newPose, data.data(), sizeof(Pose));

        // Avoid adding outdated poses
        if (!poseQueue.empty() && newPose.timestamp - poseQueue.back().timestamp <= poseDropThresUs) {
            return;
        }

        poseQueue.push_back(newPose);
        if (poseQueue.size() > maxSavedPoses) {
            poseQueue.pop_front();
        }
    }

    pose_id_t receivePose(bool setProj = true) {
        std::lock_guard<std::mutex> lock(m);

        if (poseQueue.empty()) {
            return -1;
        }

        Pose pose = poseQueue.front();
        poseQueue.pop_front();

        if (camera->isVR()) {
            auto* vrCamera = static_cast<VRCamera*>(camera);
            if (setProj) {
                vrCamera->setProjectionMatrices({pose.stereo.projL, pose.stereo.projR});
            }
            vrCamera->setViewMatrices({pose.stereo.viewL, pose.stereo.viewR});
        }
        else {
            auto* perspectiveCamera = static_cast<PerspectiveCamera*>(camera);
            if (setProj) {
                perspectiveCamera->setProjectionMatrix(pose.mono.proj);
            }
            perspectiveCamera->setViewMatrix(pose.mono.view);
        }

        return pose.id;
    }

private:
    Camera* camera;
    float poseDropThresUs;
    size_t maxSavedPoses;

    std::mutex m;
    std::deque<Pose> poseQueue;
};

} // namespace quasar

#endif // POSE_RECEIVER_H
