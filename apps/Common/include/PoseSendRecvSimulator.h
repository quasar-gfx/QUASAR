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

    struct ErrorStats {
        glm::vec2 positionErrMeanStd;
        glm::vec2 positionErrMinMax;
        glm::vec2 rotationErrMeanStd;
        glm::vec2 rotationErrMinMax;
        glm::vec2 rttMeanStd;
    };

    PoseSendRecvSimulator(double networkLatencyMs, double networkJitterMs, double renderTimeMs, unsigned int seed = 42)
            : networkLatencyS(timeutils::millisToSeconds(networkLatencyMs))
            , networkJitterS(timeutils::millisToSeconds(networkJitterMs))
            , renderTimeS(timeutils::millisToSeconds(renderTimeMs))
            , generator(seed)
            , distribution(-networkJitterS, networkJitterS) {
    }

    void setNetworkLatency(double networkLatencyMs) {
        networkLatencyS = timeutils::millisToSeconds(networkLatencyMs);
        clear();
    }

    void setNetworkJitter(double networkJitterMs) {
        networkJitterS = timeutils::millisToSeconds(networkJitterMs);
        distribution = std::uniform_real_distribution<double>(-networkJitterS, networkJitterS);
        clear();
    }

    void setRenderTime(double renderTimeMs) {
        renderTimeS = timeutils::millisToSeconds(renderTimeMs);
        clear();
    }

    void setPosePrediction(bool posePrediction) {
        this->posePrediction = posePrediction;
        clear();
    }

    void clear() {
        incomingPoses.clear();
        outPoses.clear();
        outOrigTimestamps.clear();

        positionErrors.clear();
        rotationErrors.clear();
        rtts.clear();
    }

    void sendPose(const PerspectiveCamera &camera, double now) {
        // client "sends" pose here
        incomingPoses.push_back({
                camera.getViewMatrix(),
                camera.getProjectionMatrix(),
                static_cast<uint64_t>(timeutils::secondsToMicros(now))
            });
        update(now); // update to process the incoming poses
    }

    void update(float now) {
        if (now <= lastUpdateTimeS) {
            return; // prevent multiple updates with the same or earlier time
        }
        lastUpdateTimeS = now;

        // server "receives" pose here
        if (!incomingPoses.empty()) {
            double dtFuture = networkLatencyS;

            // server waits networkLatencyS seconds before receiving the pose
            Pose poseToRecv = incomingPoses.front();
            double timestampS = timeutils::microsToSeconds(poseToRecv.timestamp);
            if (networkLatencyS > 0 && now - timestampS < dtFuture + actualInJitter) {
                return;
            }
            poseToRecv.timestamp = static_cast<uint64_t>(timeutils::secondsToMicros(now));

            // update actual jitter
            actualInJitter = randomJitter();

            outPoses.push_back(poseToRecv);
            outOrigTimestamps.push_back(timestampS);

            incomingPoses.pop_front();
        }
    }

    bool recvPoseToRender(Pose& pose, double now) {
        if (outPoses.empty() && outOrigTimestamps.empty()) {
            return false;
        }

        double dtFuture = networkLatencyS + renderTimeS;
        double jitterPredicted = randomJitter(); // we don't know the actual jitter, so we can simulate a ping

        Pose poseToSend = outPoses.front();
        if (posePrediction && outPoses.size() >= 2) {
            // get the last two poses
            auto& lastPose       = outPoses[outPoses.size() - 1];
            auto& secondLastPose = outPoses[outPoses.size() - 2];

            // predict networkLatencyS seconds into the future, accounting for jitter and render time
            if (!getPosePredicted(poseToSend, lastPose, secondLastPose, now + dtFuture + jitterPredicted)) {
                return false;
            }
        }

        // client waits for dtFuture + outJitterS before receiving the server frame and pred pose
        double timestampS = timeutils::microsToSeconds(outPoses.front().timestamp);
        if (networkLatencyS > 0 && now - timestampS < dtFuture + actualOutJitter) {
            return false;
        }

        // update actual jitter
        actualOutJitter = randomJitter();

        double oldTimestampS = outOrigTimestamps.front();
        rtts.push_back(timeutils::secondsToMillis(now - oldTimestampS));

        pose = poseToSend;
        outPoses.pop_front();
        outOrigTimestamps.pop_front();

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

    ErrorStats getAvgErrors() {
        ErrorStats stats;

        stats.positionErrMeanStd.x = calculateMean(positionErrors);
        stats.positionErrMeanStd.y = calculateStdDev(positionErrors, stats.positionErrMeanStd.x);
        stats.positionErrMinMax.x = *std::min_element(positionErrors.begin(), positionErrors.end());
        stats.positionErrMinMax.y = *std::max_element(positionErrors.begin(), positionErrors.end());

        stats.rotationErrMeanStd.x = calculateMean(rotationErrors);
        stats.rotationErrMeanStd.y = calculateStdDev(rotationErrors, stats.rotationErrMeanStd.x);
        stats.rotationErrMinMax.x = *std::min_element(rotationErrors.begin(), rotationErrors.end());
        stats.rotationErrMinMax.y = *std::max_element(rotationErrors.begin(), rotationErrors.end());

        stats.rttMeanStd.x = calculateMean(rtts);
        stats.rttMeanStd.y = calculateStdDev(rtts, stats.rttMeanStd.x);

        return stats;
    }

    void printErrors() {
        ErrorStats stats = getAvgErrors();

        spdlog::info("Pose Error:");
        spdlog::info("  Pos ({:.2f}±{:.2f},[{:.1f},{:.2f}])m", stats.positionErrMeanStd.x, stats.positionErrMeanStd.y, stats.positionErrMinMax.x, stats.positionErrMinMax.y);
        spdlog::info("  Rot ({:.2f}±{:.2f},[{:.1f},{:.2f}])°", stats.rotationErrMeanStd.x, stats.rotationErrMeanStd.y, stats.rotationErrMinMax.x, stats.rotationErrMinMax.y);
        spdlog::info("  RTT ({:.2f}±{:.2f})ms", stats.rttMeanStd.x, stats.rttMeanStd.y);
    }

private:
    double lastUpdateTimeS = -1.0;

    std::deque<Pose> incomingPoses;
    std::deque<Pose> outPoses;
    std::deque<double> outOrigTimestamps;

    std::vector<double> positionErrors;
    std::vector<double> rotationErrors;
    std::vector<double> rtts;

    double actualInJitter = randomJitter();
    double actualOutJitter = randomJitter();

    bool posePrediction = true;

    // filtering
    float alpha = 0.7f;
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
        double t1 = timeutils::microsToSeconds(secondLastPose.timestamp);
        double t0 = timeutils::microsToSeconds(lastPose.timestamp);

        float dt = t0 - t1;
        if (dt <= 0) {
            spdlog::error("Invalid timestamps for pose prediction! t0: {}, t1: {}", t0, t1);
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
        }
        else {
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
        }
        else {
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
