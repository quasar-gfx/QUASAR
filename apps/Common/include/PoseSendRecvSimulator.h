#ifndef POSE_SIM_H
#define POSE_SIM_H

#include <random>
#include <deque>
#include <vector>
#include <numeric>
#include <cmath>

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

        positionErrors.clear();
        rotationErrors.clear();
        timeErrors.clear();
    }

    void setPosePrediction(bool posePrediction) {
        this->posePrediction = posePrediction;
        clear();
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

        timeErrors.push_back(now - outTimestamps.front());

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

        positionErrors.push_back(positionDiff);
        rotationErrors.push_back(std::abs(angleDiffDegrees));
    }

    void getAvgErrors(double &positionError, double &rotationError, double &timeError,
                      double &positionStdDev, double &rotationStdDev, double &timeStdDev) {
        positionError = calculateMean(positionErrors);
        rotationError = calculateMean(rotationErrors);
        timeError = calculateMean(timeErrors);

        positionStdDev = calculateStdDev(positionErrors, positionError);
        rotationStdDev = calculateStdDev(rotationErrors, rotationError);
        timeStdDev = calculateStdDev(timeErrors, timeError);
    }

private:
    bool posePrediction = true;

    std::deque<Pose> inPoses;
    std::deque<double> inTimestamps;

    std::deque<Pose> outPoses;
    std::deque<double> outTimestamps;

    std::vector<double> positionErrors;
    std::vector<double> rotationErrors;
    std::vector<double> timeErrors;

    double randomJitter() {
        return distribution(generator);
    }

    double calculateMean(const std::vector<double>& errors) const {
        if (errors.empty()) return 0.0;
        return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    }

    double calculateStdDev(const std::vector<double>& errors, double mean) const {
        if (errors.size() < 2) return 0.0;
        double sumSquaredDiffs = 0.0;
        for (double err : errors) {
            sumSquaredDiffs += (err - mean) * (err - mean);
        }
        return std::sqrt(sumSquaredDiffs / (errors.size() - 1));
    }

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

        if (glm::dot(rotationSecondLast, rotationLatest) < 0.0f) {
            rotationSecondLast = -rotationSecondLast;
        }

        float dtFuture = targetTime - t0;

        glm::vec3 velocity = (positionLast - positionSecondLast) / dt;
        glm::vec3 predictedPosition = positionLast + velocity * dtFuture;

        glm::quat deltaRotation = glm::normalize(rotationLatest * glm::inverse(rotationSecondLast));

        float angle = glm::angle(deltaRotation);
        glm::vec3 axis = glm::axis(deltaRotation);
        if (glm::length(axis) < 1e-6f || glm::isnan(glm::length(axis))) {
            axis = glm::vec3(0.0f, 1.0f, 0.0f);
        }

        float angularVelocity = angle / dt;
        const float maxAngularVelocity = glm::radians(180.0f);
        angularVelocity = glm::clamp(angularVelocity, -maxAngularVelocity, maxAngularVelocity);

        float futureAngle = angularVelocity * dtFuture;
        glm::quat predictedRotation = glm::normalize(glm::angleAxis(futureAngle, axis) * rotationLatest);

        glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), predictedPosition) * glm::mat4_cast(predictedRotation);
        glm::mat4 predictedView = glm::inverse(predictedTransform);

        predictedPose.setViewMatrix(predictedView);
        predictedPose.setProjectionMatrix(lastPose.mono.proj);

        return true;
    }
};

#endif // POSE_SIM_H
