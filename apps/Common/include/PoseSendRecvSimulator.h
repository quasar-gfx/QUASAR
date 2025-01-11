#ifndef POSE_SIM_H
#define POSE_SIM_H

#include <random>
#include <deque>

#include <Cameras/PerspectiveCamera.h>
#include <CameraPose.h>

class PoseSendRecvSimulator {
public:
    double networkLatencyS = 0.0;
    double networkJitterS = 0.0;

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution;

    PoseSendRecvSimulator(double networkLatencyMs, double networkJitterMs, unsigned int seed = 42)
            : networkLatencyS(networkLatencyMs / MILLISECONDS_IN_SECOND)
            , networkJitterS(networkJitterMs / MILLISECONDS_IN_SECOND)
            , generator(seed)
            , distribution(-networkJitterS, networkJitterS) {
    }

    void setNetworkLatency(double networkLatencyMs) {
        this->networkLatencyS = networkLatencyMs / MILLISECONDS_IN_SECOND;
        clear();
    }

    void setNetworkJitter(double networkJitterMs) {
        this->networkJitterS = networkJitterMs / MILLISECONDS_IN_SECOND;
        distribution = std::uniform_real_distribution<double>(-networkJitterS, networkJitterS);
        clear();
    }

    void clear() {
        inPoses.clear();
        inTimestamps.clear();
        outPoses.clear();
        outTimestamps.clear();

        totalPositionError = 0.0f;
        totalRotationError = 0.0f;
        totalTimeError = 0.0f;
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

        double jitterS = randomJitter();

        // server waits networkLatencyS seconds before receiving the pose
        if (networkLatencyS != 0 && now - inTimestamps.front() <= networkLatencyS + jitterS) {
            return false;
        }

        Pose poseToSend = inPoses.front();

        if (posePrediction && (inPoses.size() >= 2 && inTimestamps.size() >= 2)) {
            // predict networkLatencyS + jitterS seconds into the future
            if (!getPosePredicted(poseToSend, now + (networkLatencyS + jitterS))) {
                return false;
            }
        }

        outPoses.push_back(poseToSend);
        outTimestamps.push_back(now);

        inPoses.pop_front();
        inTimestamps.pop_front();

        jitterS = randomJitter();

        // client waits networkLatencyS seconds before receiving the next pose
        if (networkLatencyS != 0 && now - outTimestamps.front() <= networkLatencyS + jitterS) {
            return false;
        }

        totalTimeError += now - outTimestamps.front();

        pose = outPoses.front();

        outPoses.pop_front();
        outTimestamps.pop_front();

        return true;
    }

    void accumulateError(const PerspectiveCamera &camera, const PerspectiveCamera &remoteCamera) {
        float positionDiff = glm::distance(camera.getPosition(), remoteCamera.getPosition());

        glm::quat q1 = glm::normalize(camera.getRotationQuat());
        glm::quat q2 = glm::normalize(remoteCamera.getRotationQuat());

        // ensure the shortest path for quaternion interpolation
        if (glm::dot(q1, q2) < 0.0f) {
            q2 = -q2;
        }

        float angleDiffRadians = 2.0f * glm::acos(glm::clamp(glm::dot(q1, q2), -1.0f, 1.0f));
        float angleDiffDegrees = glm::degrees(angleDiffRadians);

        totalPositionError += positionDiff;
        totalRotationError += glm::abs(angleDiffDegrees);
        count++;
    }

    void getAvgErrors(double &positionError, double &rotationError, double &timeError) {
        if (count == 0) {
            positionError = 0.0f;
            rotationError = 0.0f;
            timeError = 0.0f;
            return;
        }

        positionError = totalPositionError / count;
        rotationError = totalRotationError / count;
        timeError = totalTimeError / count;
    }

private:
    bool posePrediction = true;

    double totalPositionError = 0.0f;
    double totalRotationError = 0.0f;
    double totalTimeError = 0.0f;
    int count = 0;

    std::deque<Pose> inPoses;
    std::deque<double> inTimestamps;

    std::deque<Pose> outPoses;
    std::deque<double> outTimestamps;

    bool getPosePredicted(Pose& predictedPose, double targetTime) {
        int poseCount = inPoses.size();

        auto secondLastPose = inPoses[poseCount - 2];
        auto lastPose       = inPoses[poseCount - 1];

        double t1 = inTimestamps[poseCount - 2];
        double t0 = inTimestamps[poseCount - 1];

        float dt = t0 - t1;

        if (dt <= 0) {
            spdlog::error("Invalid timestamps for pose prediction!");
            return false;
        }

        glm::vec3 scale, skew;
        glm::vec4 perspective;
        glm::vec3 positionSecondLast, positionLast;
        glm::quat rotationSecondLast, rotationLatest;

        glm::decompose(glm::inverse(secondLastPose.mono.view), scale, rotationSecondLast, positionSecondLast, skew, perspective);
        glm::decompose(glm::inverse(lastPose.mono.view), scale, rotationLatest, positionLast, skew, perspective);

        float dtFuture = targetTime - t0;

        glm::vec3 velocity = (positionLast - positionSecondLast) / dt;
        glm::vec3 predictedPosition = positionLast + velocity * dtFuture;

        glm::quat deltaRotation = glm::normalize(rotationLatest * glm::inverse(rotationSecondLast));

        float angle;
        glm::vec3 axis;
        angle = glm::angle(deltaRotation);
        axis = glm::axis(deltaRotation);

        if (glm::length(axis) < 1e-6f) {
            axis = glm::vec3(0.0f, 1.0f, 0.0f); // default axis if undefined
        }

        float angularVelocity = angle / dt;
        float futureAngle = angularVelocity * dtFuture;
        glm::quat predictedRotation = glm::normalize(glm::angleAxis(futureAngle, axis) * rotationLatest);

        glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), predictedPosition) * glm::mat4_cast(predictedRotation);
        glm::mat4 predictedView = glm::inverse(predictedTransform);

        predictedPose.setViewMatrix(predictedView);
        predictedPose.setProjectionMatrix(lastPose.mono.proj);

        return true;
    }

    double randomJitter() {
        return distribution(generator);
    }
};

#endif // POSE_SIM_H
