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
#include <PerspectiveCamera.h>
#include <VRCamera.h>
#include <Networking/DataStreamerUDP.h>

#include <CameraPose.h>

class PoseStreamer {
public:
    std::string receiverURL;

    explicit PoseStreamer(PerspectiveCamera* camera, std::string receiverURL)
            : camera(camera)
            , receiverURL(receiverURL)
            , streamer(receiverURL, sizeof(Pose)) { }

    explicit PoseStreamer(VRCamera* vrcamera, std::string receiverURL)
            : vrcamera(vrcamera)
            , receiverURL(receiverURL)
            , streamer(receiverURL, sizeof(Pose)) { }

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
        currPose.id = currPoseID;
#ifdef VR
        currPose.projL = vrcamera->left.getProjectionMatrix();
        currPose.projR = vrcamera->right.getProjectionMatrix();
        currPose.viewL = vrcamera->left.getViewMatrix();
        currPose.viewR = vrcamera->right.getViewMatrix();
        if (epsilonEqual(currPose.viewL, prevPose.viewL) ||
            epsilonEqual(currPose.viewR, prevPose.viewR)) {
            return false;
        }
#else
        currPose.proj = camera->getProjectionMatrix();
        currPose.view = camera->getViewMatrix();
        if (epsilonEqual(currPose.view, prevPose.view)) {
            return false;
        }
#endif
        currPose.timestamp = timeutils::getTimeMillis();
        streamer.send((uint8_t*)&currPose);

        prevPoses[currPoseID] = currPose;
        currPoseID++;

        return true;
    }

private:
    DataStreamerUDP streamer;

    PerspectiveCamera* camera;
    VRCamera* vrcamera;
    Pose currPose, prevPose;
    pose_id_t currPoseID = 0;

    std::map<pose_id_t, Pose> prevPoses;
};

#endif // POSE_STREAMER_H
