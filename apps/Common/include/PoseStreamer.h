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

#include <CameraPose.h>

class PoseStreamer {
public:
    std::string receiverURL;

    SocketUDP socket;

    Camera* camera;

    Pose currPose, prevPose;
    pose_id_t currPoseID = 0;

    std::map<pose_id_t, Pose> prevPoses;

    explicit PoseStreamer(Camera* camera, std::string receiverURL)
            : camera(camera)
            , receiverURL(receiverURL)
            , socket(true) {
        socket.setAddress(receiverURL);
    }

    bool epsilonEqual(const glm::mat4& mat1, const glm::mat4& mat2, float epsilon = 0.001f) {
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
                // get unix timestamp in ms
                std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                );
                *elapsedTime = ms.count() - pose->timestamp;
            }

            // delete all poses with id less than poseID
            for (auto it = prevPoses.begin(); it != prevPoses.end();) {
                if (it->first < poseID) {
                    it = prevPoses.erase(it);
                }
                else {
                    ++it;
                }
            }

            return true;
        }

        return false;
    }

    bool sendPose() {
        currPose.id = currPoseID;
        currPose.proj = camera->getProjectionMatrix();
        currPose.view = camera->getViewMatrix();
        // get unix timestamp in ms
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        );
        currPose.timestamp = ms.count();

        if (epsilonEqual(currPose.view, prevPose.view)) {
            return false;
        }

        int bytesSent = socket.send(&currPose, sizeof(Pose), 0);
        if (bytesSent < 0) {
            return false;
        }

        prevPoses[currPoseID] = currPose;
        currPoseID++;

        // prevPose.prevViewMatrix = viewMatrix;

        return bytesSent >= 0;
    }
};

#endif // POSE_STREAMER_H
