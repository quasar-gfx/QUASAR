#ifndef POSE_SIM_H
#define POSE_SIM_H

#include <deque>

#include <Cameras/PerspectiveCamera.h>
#include <CameraPose.h>

class PoseSendRecvSimulator {
public:
    double networkLatency = 0.0;

    PoseSendRecvSimulator(double networkLatency) : networkLatency(networkLatency) {}

    void setNetworkLatency(double networkLatency) {
        this->networkLatency = networkLatency;

        // clear all inPoses
        inPoses.clear();
        inTimestamps.clear();
    }

    void setPosePrediction(bool posePrediction) {
        this->posePrediction = posePrediction;
    }

    void sendPose(const PerspectiveCamera &camera, double now) {
        inPoses.push_back({camera.getViewMatrix(), camera.getProjectionMatrix(), 0});
        inTimestamps.push_back(now);
    }

    bool recvPose(Pose& pose, double now) {
        if (inPoses.empty() || inTimestamps.empty()) {
            return false;
        }

        // server waits networkLatency ms before receiving the pose
        if (now - inTimestamps.front() < networkLatency / MILLISECONDS_IN_SECOND) {
            return false;
        }

        Pose poseToSend = inPoses.front();

        if (posePrediction) {
            getPosePredicted(poseToSend, now + networkLatency / MILLISECONDS_IN_SECOND);
        }

        outPoses.push_back(poseToSend);
        outTimestamps.push_back(now);

        inPoses.pop_front();
        inTimestamps.pop_front();

        // client waits networkLatency ms before receiving the next pose
        if (now - outTimestamps.front() < networkLatency / MILLISECONDS_IN_SECOND) {
            return false;
        }

        pose = outPoses.front();

        outPoses.pop_front();
        outTimestamps.pop_front();

        return true;
    }

private:
    bool posePrediction = true;

    std::deque<Pose> inPoses;
    std::deque<double> inTimestamps;

    std::deque<Pose> outPoses;
    std::deque<double> outTimestamps;

    bool getPosePredicted(Pose& predictedPose, double timeOfPrediction) {
        if (inPoses.size() < 3 || inTimestamps.size() < 3) {
            return false;
        }

        auto thirdLastPose  = inPoses[inPoses.size() - 3];
        auto secondLastPose = inPoses[inPoses.size() - 2];
        auto lastPose       = inPoses[inPoses.size() - 1];

        double t1 = inTimestamps[inTimestamps.size() - 3];
        double t2 = inTimestamps[inTimestamps.size() - 2];
        double t3 = inTimestamps[inTimestamps.size() - 1];

        double dt1 = t2 - t1;
        double dt2 = t3 - t2;

        if (dt1 == 0 || dt2 == 0) {
            return false;
        }

        glm::vec3 scale, skew;
        glm::vec4 perspective;
        glm::vec3 position1, position2, position3;
        glm::quat rotation1, rotation2, rotation3;

        glm::decompose(thirdLastPose.mono.view, scale, rotation1, position1, skew, perspective);
        glm::decompose(secondLastPose.mono.view, scale, rotation2, position2, skew, perspective);
        glm::decompose(lastPose.mono.view, scale, rotation3, position3, skew, perspective);

        position1 = -position1;
        position2 = -position2;
        position3 = -position3;

        rotation1 = glm::conjugate(rotation1);
        rotation2 = glm::conjugate(rotation2);
        rotation3 = glm::conjugate(rotation3);

        glm::vec3 velocity1 = (position2 - position1) / static_cast<float>(dt1);
        glm::vec3 velocity2 = (position3 - position2) / static_cast<float>(dt2);

        glm::vec3 acceleration = (velocity2 - velocity1) / static_cast<float>(dt2);

        double deltaTime = timeOfPrediction - t3;

        glm::vec3 predictedPosition = position3 + velocity2 * static_cast<float>(deltaTime) +
                                      0.5f * acceleration * static_cast<float>(deltaTime * deltaTime);

        glm::quat velocityRotation1 = glm::normalize(rotation2 * glm::inverse(rotation1));
        glm::quat velocityRotation2 = glm::normalize(rotation3 * glm::inverse(rotation2));

        glm::quat accelerationRotation = glm::normalize(velocityRotation2 * glm::inverse(velocityRotation1));

        glm::quat predictedRotation = rotation3 * glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), accelerationRotation, static_cast<float>(deltaTime));

        glm::mat4 predictedView = glm::translate(glm::mat4(1.0f), -predictedPosition) * glm::mat4_cast(glm::conjugate(predictedRotation));

        predictedPose.setViewMatrix(predictedView);
        predictedPose.setProjectionMatrix(lastPose.mono.proj);

        return true;
    }
};

#endif // POSE_SIM_H
