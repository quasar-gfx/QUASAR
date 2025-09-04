#include <spdlog/spdlog.h>

#include <Utils/TimeUtils.h>
#include <PoseSendRecvSimulator.h>

namespace quasar {

PoseSendRecvSimulator::PoseSendRecvSimulator(PoseSendRecvSimulatorCreateParams params)
    : networkLatencyS(timeutils::millisToSeconds(params.networkLatencyMs))
    , networkJitterS(timeutils::millisToSeconds(params.networkJitterMs))
    , renderTimeS(timeutils::millisToSeconds(params.renderTimeMs))
    , posePrediction(params.posePrediction)
    , poseSmoothing(params.poseSmoothing)
    , generator(params.seed)
    , distribution(-networkJitterS, networkJitterS)
    , actualInJitter(randomJitter())
    , actualOutJitter(randomJitter())
{}

void PoseSendRecvSimulator::setNetworkLatency(double networkLatencyMs) {
    networkLatencyS = timeutils::millisToSeconds(networkLatencyMs);
    clear();
}

void PoseSendRecvSimulator::setNetworkJitter(double networkJitterMs) {
    networkJitterS = timeutils::millisToSeconds(networkJitterMs);
    distribution = std::uniform_real_distribution<double>(-networkJitterS, networkJitterS);
    clear();
}

void PoseSendRecvSimulator::setRenderTime(double renderTimeMs) {
    renderTimeS = timeutils::millisToSeconds(renderTimeMs);
    clear();
}

void PoseSendRecvSimulator::clear() {
    incomingPoses.clear();
    outPoses.clear();
    outOrigTimestamps.clear();
    positionErrors.clear();
    rotationErrors.clear();
    rtts.clear();
    positionHistory.clear();
    rotationHistory.clear();
}

void PoseSendRecvSimulator::sendPose(const PerspectiveCamera& camera, double now) {
    incomingPoses.push_back({
        camera.getViewMatrix(),
        camera.getProjectionMatrix(),
        static_cast<time_t>(timeutils::secondsToMicros(now))
    });
    update(now);
}

void PoseSendRecvSimulator::update(float now) {
    if (now <= lastUpdateTimeS) return;
    lastUpdateTimeS = now;

    if (!incomingPoses.empty()) {
        double dtFuture = networkLatencyS;
        Pose poseToRecv = incomingPoses.front();
        double timestampS = timeutils::microsToSeconds(poseToRecv.timestamp);
        if (networkLatencyS > 0 && now - timestampS < dtFuture + actualInJitter) return;

        poseToRecv.timestamp = static_cast<time_t>(timeutils::secondsToMicros(now));
        actualInJitter = randomJitter();

        outPoses.push_back(poseToRecv);
        outOrigTimestamps.push_back(timestampS);
        incomingPoses.pop_front();
    }
}

bool PoseSendRecvSimulator::recvPoseToRender(Pose& pose, double now) {
    if (outPoses.empty() && outOrigTimestamps.empty()) return false;

    double dtFuture = networkLatencyS + renderTimeS;
    double jitterPredicted = randomJitter();

    Pose poseToSend = (networkLatencyS != 0) ? outPoses.front() : outPoses.back();
    if (posePrediction && outPoses.size() >= 3) {
        auto& lastPose = outPoses[outPoses.size() - 1];
        auto& prevPose = outPoses[outPoses.size() - 2];
        auto& secondPrevPose = outPoses[outPoses.size() - 3];

        if (!getPosePredicted(poseToSend, lastPose, prevPose, secondPrevPose, now + dtFuture + jitterPredicted)) {
            return false;
        }
    }

    double timestampS = timeutils::microsToSeconds(outPoses.front().timestamp);
    if (networkLatencyS > 0 && now - timestampS < dtFuture + actualOutJitter) return false;

    actualOutJitter = randomJitter();

    double oldTimestampS = outOrigTimestamps.front();
    rtts.push_back(timeutils::secondsToMillis(now - oldTimestampS));

    pose = poseToSend;
    if (!posePrediction || outPoses.size() >= 3) {
        outPoses.pop_front();
        outOrigTimestamps.pop_front();
    }

    return true;
}

void PoseSendRecvSimulator::accumulateError(const PerspectiveCamera& camera, const PerspectiveCamera& remoteCamera) {
    float positionDiff = glm::distance(camera.getPosition(), remoteCamera.getPosition());
    glm::quat q1 = glm::normalize(camera.getRotationQuat());
    glm::quat q2 = glm::normalize(remoteCamera.getRotationQuat());
    if (glm::dot(q1, q2) < 0.0f) q2 = -q2;

    float angleDiffRadians = 2.0f * glm::acos(glm::clamp(glm::dot(q1, q2), -1.0f, 1.0f));
    float angleDiffDegrees = glm::degrees(angleDiffRadians);

    positionErrors.push_back(positionDiff);
    rotationErrors.push_back(std::abs(angleDiffDegrees));
}

PoseSendRecvSimulator::ErrorStats PoseSendRecvSimulator::getAvgErrors() {
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

void PoseSendRecvSimulator::printErrors() {
    ErrorStats stats = getAvgErrors();
    spdlog::info("Pose Error:");
    spdlog::info("  Pos ({:.2f}±{:.2f},[{:.1f},{:.2f}])m", stats.positionErrMeanStd.x, stats.positionErrMeanStd.y, stats.positionErrMinMax.x, stats.positionErrMinMax.y);
    spdlog::info("  Rot ({:.2f}±{:.2f},[{:.1f},{:.2f}])°", stats.rotationErrMeanStd.x, stats.rotationErrMeanStd.y, stats.rotationErrMinMax.x, stats.rotationErrMinMax.y);
    spdlog::info("  RTT ({:.2f}±{:.2f})ms", stats.rttMeanStd.x, stats.rttMeanStd.y);
}

glm::vec3 PoseSendRecvSimulator::savitzkyGolayFilter(const std::deque<glm::vec3>& buffer) {
    if (buffer.size() < 5) return buffer.back();
    static const std::array<float, 5> coeffs = {
        -3.0f / 35.0f, 12.0f / 35.0f, 17.0f / 35.0f, 12.0f / 35.0f, -3.0f / 35.0f
    };
    glm::vec3 result(0.0f);
    for (int i = 0; i < 5; i++) {
        result += coeffs[i] * buffer[buffer.size() - 5 + i];
    }
    return result;
}

glm::quat PoseSendRecvSimulator::averageQuaternions(const std::deque<glm::quat>& quats) {
    if (quats.empty()) return glm::quat(1, 0, 0, 0);
    glm::quat avg = quats[0];
    for (size_t i = 1; i < quats.size(); i++) {
        if (glm::dot(avg, quats[i]) < 0.0f)
            avg = glm::slerp(avg, -quats[i], 1.0f / (i + 1));
        else
            avg = glm::slerp(avg, quats[i], 1.0f / (i + 1));
    }
    return glm::normalize(avg);
}

double PoseSendRecvSimulator::randomJitter() {
    return distribution(generator);
}

double PoseSendRecvSimulator::calculateMean(const std::vector<double>& errors) const {
    if (errors.empty()) return 0.0;
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

double PoseSendRecvSimulator::calculateStdDev(const std::vector<double>& errors, double mean) const {
    if (errors.size() < 2) return 0.0;
    double sumSquaredDiffs = 0.0;
    for (double err : errors) {
        sumSquaredDiffs += (err - mean) * (err - mean);
    }
    return std::sqrt(sumSquaredDiffs / (errors.size() - 1));
}

bool PoseSendRecvSimulator::getPosePredicted(
    Pose& predictedPose,
    const Pose& latest, const Pose& previous, const Pose& secondPrevious,
    double targetFutureTimeS)
{
    double t2 = timeutils::microsToSeconds(secondPrevious.timestamp);
    double t1 = timeutils::microsToSeconds(previous.timestamp);
    double t0 = timeutils::microsToSeconds(latest.timestamp);

    float dt1 = t1 - t2;
    float dt2 = t0 - t1;

    if (dt1 <= 0.0f || dt2 <= 0.0f) return false;

    float dtFuture = targetFutureTimeS - t0;
    const float maxPredictTime = 0.1f;
    dtFuture = glm::clamp(dtFuture, 0.0f, maxPredictTime);

    glm::vec3 scale, skew;
    glm::vec4 perspective;

    glm::vec3 p2, p1, p0;
    glm::quat r2, r1, r0;

    glm::decompose(glm::inverse(secondPrevious.mono.view), scale, r2, p2, skew, perspective);
    glm::decompose(glm::inverse(previous.mono.view), scale, r1, p1, skew, perspective);
    glm::decompose(glm::inverse(latest.mono.view), scale, r0, p0, skew, perspective);

    if (glm::dot(r1, r0) < 0.0f) r1 = -r1;
    if (glm::dot(r2, r1) < 0.0f) r2 = -r2;

    glm::vec3 filteredP0 = poseSmoothing ? savitzkyGolayFilter([&] {
        positionHistory.push_back(p0);
        if (positionHistory.size() > maxPositionHistorySize) positionHistory.pop_front();
        return positionHistory;
    }()) : p0;

    glm::vec3 v1 = (p1 - p2) / dt1;
    glm::vec3 v2 = (p0 - p1) / dt2;
    glm::vec3 v = 0.5f * (v1 + v2);
    glm::vec3 a = (v2 - v1) / dt2;
    a = glm::clamp(a, -3.0f, 3.0f);

    glm::vec3 rawPrediction = filteredP0 + v * dtFuture + 0.5f * a * dtFuture * dtFuture;

    float confidence = 1.0f - glm::smoothstep(0.02f, 0.06f, dtFuture);
    glm::vec3 finalPrediction = poseSmoothing ? glm::mix(filteredP0, rawPrediction, confidence) : rawPrediction;

    glm::quat dq = glm::normalize(r0 * glm::inverse(r1));
    float angle = glm::angle(dq);
    glm::vec3 axis = glm::axis(dq);
    if (glm::length(axis) < 1e-5f || glm::any(glm::isnan(axis))) axis = glm::vec3(0, 1, 0);
    float angularSpeed = angle / dt2;

    angularSpeed = glm::clamp(angularSpeed, 0.0f, glm::radians(200.0f));
    float futureAngle = angularSpeed * dtFuture;
    futureAngle = glm::clamp(futureAngle, 0.0f, glm::radians(45.0f));

    glm::quat deltaFuture = glm::angleAxis(futureAngle, axis);
    glm::quat predictedRotation = glm::normalize(deltaFuture * r0);

    glm::quat finalRotation = poseSmoothing ? averageQuaternions([&] {
        rotationHistory.push_back(predictedRotation);
        if (rotationHistory.size() > maxRotationHistorySize) rotationHistory.pop_front();
        return rotationHistory;
    }()) : predictedRotation;

    glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), finalPrediction) * glm::mat4_cast(finalRotation);
    glm::mat4 predictedView = glm::inverse(predictedTransform);

    predictedPose.setViewMatrix(predictedView);
    predictedPose.setProjectionMatrix(latest.mono.proj);

    return true;
}

} // namespace quasar
