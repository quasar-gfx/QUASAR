#ifndef POSE_SIM_H
#define POSE_SIM_H

#include <random>
#include <deque>
#include <vector>
#include <numeric>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>

#include <Windowing/Window.h>
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
        incomingPoses.clear();
        incomingOrigTimestamps.clear();
        storedPoses.clear();
        storedOrigTimestamps.clear();
        outPoses.clear();
        outOrigTimestamps.clear();

        positionErrors.clear();
        rotationErrors.clear();
        rttErrors.clear();
    }

    void sendPose(const PerspectiveCamera &camera, double now) {
        // client "sends" the pose here!
        incomingPoses.push_back({
                camera.getViewMatrix(),
                camera.getProjectionMatrix(),
                static_cast<uint64_t>(now * MICROSECONDS_IN_SECOND)
            });
        incomingOrigTimestamps.push_back(now);

        update(now);
    }

    void update(double now) {
        // if there are incoming poses, "receive" them at a delay
        while (!incomingPoses.empty()) {
            double dtFuture = networkLatencyS + actualInJitter;

            Pose poseToRecv = incomingPoses.front();
            double timestampS = timeutils::microsToSeconds(poseToRecv.timestamp);

            // server waits networkLatencyS seconds before receiving the pose
            if (networkLatencyS > 0 && now - timestampS < dtFuture) {
                break;
            }
            poseToRecv.timestamp = static_cast<uint64_t>(timeutils::secondsToMicros(now));

            // update actual jitter
            actualInJitter = randomJitter();

            // server "recieves" the pose here! do pose prediction if needed
            Pose poseToSend = poseToRecv;
            if (posePrediction && incomingPoses.size() >= 2) {
                double jitterPredicted = randomJitter(); // we don't know the actual jitter, so we can simulate a ping
                double dtFuturePredicted = networkLatencyS + renderTimeS + jitterPredicted;

                if (!getPosePredicted(poseToSend, poseToRecv, incomingPoses.front(), now + dtFuturePredicted)) {
                    break;
                }
            }

            storedPoses.push_back(poseToSend);
            storedOrigTimestamps.push_back(timestampS);

            incomingPoses.pop_front();
            incomingOrigTimestamps.pop_front();
        }

        while (!storedPoses.empty()) {
            double dtFuture = networkLatencyS + renderTimeS + actualOutJitter;

            Pose poseToSend = storedPoses.front();
            double timestampS = timeutils::microsToSeconds(poseToSend.timestamp);

            // server "sends" the pose here!
            // client waits networkLatencyS + renderTimeS + actualOutJitter before receiving the pose
            if (networkLatencyS > 0 && now - timestampS < dtFuture) {
                break;
            }
            poseToSend.timestamp = static_cast<uint64_t>(timeutils::secondsToMicros(now));

            // update actual jitter
            actualOutJitter = randomJitter();

            outPoses.push_back(poseToSend);
            outOrigTimestamps.push_back(storedOrigTimestamps.front());

            storedPoses.pop_front();
            storedOrigTimestamps.pop_front();
        }
    }

    bool recvPose(Pose& poseToRecv, double now) {
        if (outPoses.empty()) {
            return false;
        }

        // client "receives" the pose here! just return the latest one and clear the rest
        poseToRecv = outPoses.back();

        double origTimestampS = outOrigTimestamps.back();
        rttErrors.push_back(timeutils::secondsToMillis(now - origTimestampS));

        outPoses.clear();
        outOrigTimestamps.clear();

        return true;
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

        stats.rttMeanStd.x = calculateMean(rttErrors);
        stats.rttMeanStd.y = calculateStdDev(rttErrors, stats.rttMeanStd.x);

        return stats;
    }

    void accumulateErrors(const PerspectiveCamera &camera, const PerspectiveCamera &remoteCamera) {
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

    void printErrors() {
        ErrorStats stats = getAvgErrors();

        spdlog::info("Pose Error:");
        spdlog::info("  Pos ({:.2f}±{:.2f},[{:.1f},{:.2f}])m", stats.positionErrMeanStd.x, stats.positionErrMeanStd.y, stats.positionErrMinMax.x, stats.positionErrMinMax.y);
        spdlog::info("  Rot ({:.2f}±{:.2f},[{:.1f},{:.2f}])°", stats.rotationErrMeanStd.x, stats.rotationErrMeanStd.y, stats.rotationErrMinMax.x, stats.rotationErrMinMax.y);
        spdlog::info("  RTT ({:.2f}±{:.2f})ms", stats.rttMeanStd.x, stats.rttMeanStd.y);
    }

private:
    std::deque<Pose> incomingPoses;
    std::deque<double> incomingOrigTimestamps;
    std::deque<Pose> storedPoses;
    std::deque<double> storedOrigTimestamps;
    std::deque<Pose> outPoses;
    std::deque<double> outOrigTimestamps;

    std::vector<double> positionErrors;
    std::vector<double> rotationErrors;
    std::vector<double> rttErrors;

    double actualInJitter = randomJitter();
    double actualOutJitter = randomJitter();

    bool posePrediction = true;

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
            spdlog::error("Invalid timestamps for pose prediction (t0: {}, t1: {})", t0, t1);
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
