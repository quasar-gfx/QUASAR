#ifndef POSE_STREAMER_H
#define POSE_STREAMER_H

#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <map>

#include <Socket.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/epsilon.hpp>

#include <Camera.h>
#include <DataStreamer.h>

#include <CameraPose.h>

class PoseStreamer {
public:
    std::string receiverURL;

    explicit PoseStreamer(Camera* camera, std::string receiverURL)
            : camera(camera)
            , receiverURL(receiverURL)
            , streamer(receiverURL, sizeof(Pose)) { }

    bool epsilonEqual(const glm::mat4& mat1, const glm::mat4& mat2, float epsilon = 0.001f) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(mat1[i][j] - mat2[i][j]) > epsilon)
                    return false;
            }
        }
        return true;
    }

    int getCurrTimeMillis() {
        // get unix timestamp in ms
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        );
        return ms.count();
    }

    bool getPose(pose_id_t poseID, Pose* pose, double* elapsedTime = nullptr) {
        auto res = prevPoses.find(poseID);
        if (res != prevPoses.end()) { // found
            *pose = res->second;
            if (elapsedTime) {
                *elapsedTime = getCurrTimeMillis() - pose->timestamp;
            }

            return true;
        }

        return false;
    }

    void removePosesLessThan(pose_id_t poseID) {
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
        currPose.proj = camera->getProjectionMatrix();
        currPose.view = camera->getViewMatrix();
        currPose.timestamp = getCurrTimeMillis();

        if (epsilonEqual(currPose.view, prevPose.view)) {
            return false;
        }

        streamer.send((uint8_t*)&currPose);

        prevPoses[currPoseID] = currPose;
        currPoseID++;

        return true;
    }

private:
    DataStreamerUDP streamer;

    Camera* camera;

    Pose currPose, prevPose;
    pose_id_t currPoseID = 0;

    std::map<pose_id_t, Pose> prevPoses;
};

#endif // POSE_STREAMER_H
