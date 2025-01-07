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

        // clear avg errors
        totalPositionError = 0.0f;
        totalRotationError = 0.0f;
        count = 0;
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
        if (networkLatency != 0 && now - inTimestamps.front() <= networkLatency / MILLISECONDS_IN_SECOND) {
            return false;
        }

        Pose poseToSend = inPoses.front();

        if (posePrediction && (inPoses.size() >= 3 && inTimestamps.size() >= 3)) {
            // predict networkLatency ms into the future
            if (!getPosePredicted(poseToSend, now + networkLatency / MILLISECONDS_IN_SECOND)) {
                return false;
            }
        }

        outPoses.push_back(poseToSend);
        outTimestamps.push_back(now);

        inPoses.pop_front();
        inTimestamps.pop_front();

        // client waits networkLatency ms before receiving the next pose
        if (networkLatency != 0 && now - outTimestamps.front() <= networkLatency / MILLISECONDS_IN_SECOND) {
            return false;
        }

        pose = outPoses.front();

        outPoses.pop_front();
        outTimestamps.pop_front();

        return true;
    }

    void accumulateError(const PerspectiveCamera &camera, const PerspectiveCamera &remoteCamera) {
        float positionDiff = glm::distance(camera.getPosition(), remoteCamera.getPosition());

        glm::quat delta = glm::conjugate(camera.getRotationQuat()) * remoteCamera.getRotationQuat();
        float angleDiffRadians = 2.0f * glm::acos(glm::clamp(delta.w, -1.0f, 1.0f));
        float angleDiffDegrees = glm::degrees(angleDiffRadians);
        if (angleDiffDegrees > 180.0f) {
            angleDiffDegrees -= 360.0f;
        }
        else if (angleDiffDegrees < -180.0f) {
            angleDiffDegrees += 360.0f;
        }

        totalPositionError += positionDiff;
        totalRotationError += glm::abs(angleDiffDegrees);
        count++;
    }

    void getAvgErrors(float &positionError, float &rotationError) {
        if (count == 0) {
            positionError = 0.0f;
            rotationError = 0.0f;
            return;
        }
        positionError = totalPositionError / count;
        rotationError = totalRotationError / count;
    }

private:
    bool posePrediction = false;

    float totalPositionError = 0.0f;
    float totalRotationError = 0.0f;
    int count = 0;

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

        float dt1 = t2 - t1;
        float dt2 = t3 - t2;

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

        float dt = targetTime - t3;

        glm::vec3 velocity1 = (position2 - position1) / dt1;
        glm::vec3 velocity2 = (position3 - position2) / dt2;
        glm::vec3 linearAcc = (velocity2 - velocity1) / dt2;

        // p = p0 + v0*t + 0.5*a*t^2
        glm::vec3 predictedPosition = position3 +
                                      velocity2 * dt +
                                      0.5f * linearAcc * (dt * dt);

        glm::quat delta1 = glm::conjugate(rotation1) * rotation2;
        glm::quat delta2 = glm::conjugate(rotation2) * rotation3;

        // glm::vec3 angVelocity1 = glm::axis(delta1) * glm::angle(delta1) / dt1;
        // glm::vec3 angVelocity2 = glm::axis(delta2) * glm::angle(delta2) / dt2;
        // glm::vec3 angAcc = (angVelocity2 - angVelocity1) / dt2;
        // angAcc = glm::clamp(angAcc, glm::vec3(-0.1f), glm::vec3(0.1f));

        // glm::quat angDisplacement = angVelocity2 * dt + 0.5f * angAcc * (dt * dt);

        // float futureAngle = glm::angle(angDisplacement);
        // glm::vec3 futureAxis = glm::axis(angDisplacement);

        // glm::quat rotationDelta = glm::angleAxis(futureAngle, futureAxis);
        // glm::quat predictedRotation = glm::normalize(rotationDelta * rotation3);
        glm::quat predictedRotation = rotation3;

        glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), predictedPosition) * glm::mat4_cast(predictedRotation);
        glm::mat4 predictedView = glm::inverse(predictedTransform);

        predictedPose.setViewMatrix(predictedView);
        predictedPose.setProjectionMatrix(lastPose.mono.proj);

        return true;
    }
};

#endif // POSE_SIM_H
