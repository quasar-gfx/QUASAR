#ifndef POSE_STREAMER_H
#define POSE_STREAMER_H

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <map>

#include <Networking/Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Utils/TimeUtils.h>
#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>
#include <Networking/DataStreamerUDP.h>

#include <CameraPose.h>

class PoseStreamer {
public:
    std::string receiverURL;
    unsigned int rate;

    PoseStreamer(Camera* camera, std::string receiverURL, unsigned int rate = 60)
            : camera(camera)
            , receiverURL(receiverURL)
            , rate(rate)
            , streamer(receiverURL, sizeof(Pose)) {
        std::cout << "Created PoseStreamer that sends to URL: " << receiverURL << std::endl;
    }

    bool epsilonEqual(const glm::mat4& mat1, const glm::mat4& mat2, float epsilon = 1e-5) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(mat1[i][j] - mat2[i][j]) > epsilon)
                    return false;
            }
        }
        return true;
    }

    bool getPose(pose_id_t poseID, Pose* pose, double* elapsedTime = nullptr) {
        auto res = prevPoses.find(poseID);
        if (res != prevPoses.end()) { // found
            *pose = res->second;
            if (elapsedTime) {
                *elapsedTime = timeutils::getTimeMillis() - pose->timestamp;
            }

            return true;
        }

        return false;
    }

    void removePosesLessThan(pose_id_t poseID) {
        if (poseID == -1) {
            return;
        }

        for (auto it = prevPoses.begin(); it != prevPoses.end();) {
            if (it->first < poseID) {
                it = prevPoses.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    bool sendPose() {
        static float lastSendTime = timeutils::getTimeMicros();
        if (timeutils::getTimeMicros() - lastSendTime < MICROSECONDS_IN_SECOND / rate) {
            return false;
        }
        lastSendTime = timeutils::getTimeMicros();

        Pose currPose;
        currPose.id = currPoseID;
        if (camera->isVR()) {
            auto* vrCamera = static_cast<VRCamera*>(camera);
            currPose.setProjectionMatrices({vrCamera->left.getProjectionMatrix(), vrCamera->right.getProjectionMatrix()});
            currPose.setViewMatrices({vrCamera->left.getViewMatrix(), vrCamera->right.getViewMatrix()});
            // if (epsilonEqual(currPose.stereo.viewL, prevPose.stereo.viewL) &&
            //     epsilonEqual(currPose.stereo.viewR, prevPose.stereo.viewR)) {
            //     return false;
            // }
        }
        else {
            auto* perspectiveCamera = static_cast<PerspectiveCamera*>(camera);
            currPose.setProjectionMatrices({perspectiveCamera->getProjectionMatrix(), perspectiveCamera->getProjectionMatrix()});
            currPose.setViewMatrices({perspectiveCamera->getViewMatrix(), perspectiveCamera->getViewMatrix()});
            // if (epsilonEqual(currPose.mono.view, prevPose.mono.view)) {
            //     return false;
            // }
        }
        currPose.timestamp = timeutils::getTimeMillis();
        streamer.send((uint8_t*)&currPose);

        prevPoses[currPoseID] = currPose;
        currPoseID++;

        prevPose = currPose;

        return true;
    }

private:
    DataStreamerUDP streamer;

    Camera* camera;
    Pose prevPose;
    pose_id_t currPoseID = 0;

    std::map<pose_id_t, Pose> prevPoses;
};

#endif // POSE_STREAMER_H
