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
    double networkLatencyS;
    double networkJitterS;
    double renderTimeS;

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution;

    PoseSendRecvSimulator(double networkLatencyMs, double networkJitterMs, double renderTimeMs, unsigned int seed = 42)
            : networkLatencyS(networkLatencyMs / MILLISECONDS_IN_SECOND)
            , networkJitterS(networkJitterMs / MILLISECONDS_IN_SECOND)
            , renderTimeS(renderTimeMs / MILLISECONDS_IN_SECOND)
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

    void setRenderTime(double renderTimeMs) {
        this->renderTimeS = renderTimeMs / MILLISECONDS_IN_SECOND;
        clear();
    }

    void setPosePrediction(bool posePrediction) {
        this->posePrediction = posePrediction;
        clear();
    }

    void clear() {
        inPoses.clear();
        outPoses.clear();
        outOrigTimestamps.clear();

        positionErrors.clear();
        rotationErrors.clear();
        timeErrors.clear();
    }

    void sendPose(const PerspectiveCamera &camera, double now) {
        inPoses.push_back({camera.getViewMatrix(), camera.getProjectionMatrix(), static_cast<uint64_t>(now * MICROSECONDS_IN_SECOND)});
    }

    void update(float now) {
        // if there are incoming poses, "receive" them at a delay
        if (!inPoses.empty()) {
            double inJitterS = randomJitter();
            double futureTimeDeltaS = networkLatencyS + inJitterS;

            // server waits networkLatencyS seconds before receiving the pose
            Pose poseToRecv = inPoses.front();
            double timestampS = static_cast<double>(poseToRecv.timestamp) / MICROSECONDS_IN_SECOND;
            if (networkLatencyS != 0 && now - timestampS <= futureTimeDeltaS) {
                return;
            }

            outOrigTimestamps.push_back(timestampS);
            poseToRecv.timestamp = static_cast<uint64_t>(now * MICROSECONDS_IN_SECOND);
            outPoses.push_back(poseToRecv);

            inPoses.pop_front();
        }
    }

    bool recvPose(Pose& pose, double now) {
        // if there are poses to send, "send" them first at a delay
        if (!outPoses.empty() && !outOrigTimestamps.empty()) {
            double precictedOutJitter = randomJitter();  // we don't know the actual jitter, so we can simulate a ping
            double futureTimeDeltaS = networkLatencyS + renderTimeS;

            Pose poseToSend = outPoses.front();
            if (posePrediction && outPoses.size() >= 2) {
                // get the last two poses
                auto& secondLastPose = outPoses[outPoses.size() - 2];
                auto& lastPose       = outPoses[outPoses.size() - 1];

                // predict networkLatencyS seconds into the future, acconting for jitter and render time
                if (!getPosePredicted(poseToSend, lastPose, secondLastPose, now + futureTimeDeltaS + precictedOutJitter)) {
                    return false;
                }
            }

            // client waits for futureTimeDeltaS + outJitterS before receiving the pose
            double actualOutJitterS = randomJitter();
            double timestampS = static_cast<double>(outPoses.front().timestamp) / MICROSECONDS_IN_SECOND;
            if (networkLatencyS != 0 && now - timestampS <= futureTimeDeltaS + actualOutJitterS) {
                return false;
            }

            double oldTimestampS = outOrigTimestamps.front();
            timeErrors.push_back((now - oldTimestampS) * MILLISECONDS_IN_SECOND);

            pose = poseToSend;
            outPoses.pop_front();
            outOrigTimestamps.pop_front();

            return true;
        }

        return false;
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
    std::deque<Pose> inPoses;
    std::deque<Pose> outPoses;
    std::deque<double> outOrigTimestamps;

    std::vector<double> positionErrors;
    std::vector<double> rotationErrors;
    std::vector<double> timeErrors;

    bool posePrediction = true;

    // filtering
    float alpha = 0.6f;
    glm::vec3 filteredPosition;
    glm::quat filteredRotation;
    bool filteredPositionInitialized = false;
    bool filteredRotationInitialized = false;

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

    bool getPosePredicted(Pose& predictedPose, const Pose& lastPose, const Pose& secondLastPose, double targetFutureTimeS) {
        double t1 = static_cast<double>(secondLastPose.timestamp) / MICROSECONDS_IN_SECOND;
        double t0 = static_cast<double>(lastPose.timestamp) / MICROSECONDS_IN_SECOND;

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

        float dtFuture = targetFutureTimeS - t0;

        glm::vec3 velocity = (positionLast - positionSecondLast) / dt;
        glm::vec3 predictedPosition = positionLast + velocity * dtFuture;

        if (!filteredPositionInitialized) {
            filteredPosition = predictedPosition;
            filteredPositionInitialized = true;
        } else {
            filteredPosition = alpha * predictedPosition + (1.0f - alpha) * filteredPosition;
        }

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

        if (!filteredRotationInitialized) {
            filteredRotation = predictedRotation;
            filteredRotationInitialized = true;
        } else {
            filteredRotation = glm::slerp(filteredRotation, predictedRotation, static_cast<float>(alpha));
        }

        glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), filteredPosition) * glm::mat4_cast(filteredRotation);
        glm::mat4 predictedView = glm::inverse(predictedTransform);

        predictedPose.setViewMatrix(predictedView);
        predictedPose.setProjectionMatrix(lastPose.mono.proj);

        return true;
    }
};

#endif // POSE_SIM_H
