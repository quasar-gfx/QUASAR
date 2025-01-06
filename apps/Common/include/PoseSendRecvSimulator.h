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

        if (posePrediction && (inPoses.size() >= 3 && inTimestamps.size() >= 3)) {
            if (!getPosePredicted(poseToSend, now + networkLatency / MILLISECONDS_IN_SECOND)) {
                return false;
            }
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

    bool getPosePredicted(Pose& predictedPose, double targetTime) {
        auto thirdLastPose  = inPoses[2];
        auto secondLastPose = inPoses[1];
        auto lastPose       = inPoses[0];

        double t1 = inTimestamps[2];
        double t2 = inTimestamps[1];
        double t3 = inTimestamps[0];

        double dt1 = t2 - t1;
        double dt2 = t3 - t2;

        if (dt1 == 0 || dt2 == 0) {
            return false;
        }

        glm::vec3 scale, skew;
        glm::vec4 perspective;
        glm::vec3 position1, position2, position3;
        glm::quat rotation1, rotation2, rotation3;

        glm::decompose(glm::inverse(thirdLastPose.mono.view), scale, rotation1, position1, skew, perspective);
        glm::decompose(glm::inverse(secondLastPose.mono.view), scale, rotation2, position2, skew, perspective);
        glm::decompose(glm::inverse(lastPose.mono.view), scale, rotation3, position3, skew, perspective);

        double dt = targetTime - t3;

        glm::vec3 velocity1 = (position2 - position1) / static_cast<float>(dt1);
        glm::vec3 velocity2 = (position3 - position2) / static_cast<float>(dt2);
        glm::vec3 linearAcc = (velocity2 - velocity1) / static_cast<float>(dt2);

        // p = p0 + v0*t + 0.5*a*t^2
        glm::vec3 predictedPosition = position3 +
                                      velocity2 * static_cast<float>(dt) +
                                      0.5f * linearAcc * static_cast<float>(dt * dt);

        glm::quat deltaRotation1 = rotation2 * glm::inverse(rotation1);
        glm::quat deltaRotation2 = rotation3 * glm::inverse(rotation2);

        glm::vec3 angVelocity1 = glm::axis(deltaRotation1) * glm::angle(deltaRotation1) / static_cast<float>(dt1);
        glm::vec3 angVelocity2 = glm::axis(deltaRotation2) * glm::angle(deltaRotation2) / static_cast<float>(dt2);

        glm::vec3 angAcc = (angVelocity2 - angVelocity1) / static_cast<float>(dt2);

        glm::quat predictedRotation = rotation3;// *
                                    //   glm::quat(angVelocity2 * static_cast<float>(dt)) *
                                    //   glm::quat(0.5f * angAcc * static_cast<float>(dt * dt));

        glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), predictedPosition) * glm::mat4_cast(predictedRotation);
        glm::mat4 predictedView = glm::inverse(predictedTransform);

        predictedPose.setViewMatrix(predictedView);
        predictedPose.setProjectionMatrix(lastPose.mono.proj);

        return true;
    }
};

#endif // POSE_SIM_H
