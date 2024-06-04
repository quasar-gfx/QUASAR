#ifndef POSE_STREAMER_H
#define POSE_STREAMER_H

#include <iostream>
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
    pose_id_t currPoseId = 0;

    std::map<pose_id_t, Pose> prevPoses;

    explicit PoseStreamer(Camera* camera, std::string receiverURL) : camera(camera), receiverURL(receiverURL), socket(true) {
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

    bool getPose(pose_id_t poseId, Pose* pose, double now, double* elapsedTime = nullptr) {
        auto res = prevPoses.find(poseId);
        if (res != prevPoses.end()) { // found
            *pose = res->second;
            if (elapsedTime) {
                *elapsedTime = now - pose->timestamp;
            }

            // delete all poses with id less than poseId
            for (auto it = prevPoses.begin(); it != prevPoses.end();) {
                if (it->first < poseId) {
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

    bool sendPose(double now) {
        currPose.id = currPoseId;
        currPose.proj = camera->getProjectionMatrix();
        currPose.view = camera->getViewMatrix();
        currPose.timestamp = now;

        // if (epsilonEqual(currPose.viewMatrix, prevPose.viewMatrix)) {
        //     return;
        // }

        int bytesSent = socket.send(&currPose, sizeof(Pose), 0);
        if (bytesSent < 0) {
            return false;
        }

        prevPoses[currPoseId] = currPose;
        currPoseId++;

        // prevPose.prevViewMatrix = viewMatrix;

        return bytesSent >= 0;
    }
};

#endif // POSE_STREAMER_H
