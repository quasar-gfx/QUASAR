#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <PosePredictor.h>
#include <Utils/TimeUtils.h>

using namespace quasar;

PosePredictor::PosePredictor(PosePredictorCreateParams params)
    : enablePrediction(params.enablePrediction)
    , enableSmoothing(params.enableSmoothing)
{}

void PosePredictor::addPose(const Pose& pose) {
    poseHistory.push_back(pose);
    if (poseHistory.size() > 10) {
        poseHistory.pop_front();
    }
}

glm::vec3 PosePredictor::savitzkyGolayFilter(const std::deque<glm::vec3>& buffer) {
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

glm::quat PosePredictor::averageQuaternions(const std::deque<glm::quat>& quats) {
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

bool PosePredictor::predictPose(Pose& predictedPose, double targetFutureTimeS) {
    if (poseHistory.size() < 3) {
        if (!poseHistory.empty()) {
            predictedPose = poseHistory.back();
        }
        return false;
    }

    return predictPose(predictedPose,
                       poseHistory[poseHistory.size()-1],
                       poseHistory[poseHistory.size()-2],
                       poseHistory[poseHistory.size()-3],
                       targetFutureTimeS);
}

bool PosePredictor::predictPose(
    Pose& predictedPose,
    const Pose& latest, const Pose& previous, const Pose& secondPrevious,
    double targetFutureTimeS)
{
    if (!enablePrediction) {
        predictedPose = latest;
        return true;
    }

    double t2 = timeutils::microsToSeconds(secondPrevious.timestamp);
    double t1 = timeutils::microsToSeconds(previous.timestamp);
    double t0 = timeutils::microsToSeconds(latest.timestamp);

    float dt1 = t1 - t2;
    float dt2 = t0 - t1;

    if (dt1 <= 1e-5f || dt2 <= 1e-5f) return false;

    float dtFuture = static_cast<float>(targetFutureTimeS - t0);
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

    glm::vec3 filteredP0 = enableSmoothing ? savitzkyGolayFilter([&] {
        positionSmoothingHistory.push_back(p0);
        if (positionSmoothingHistory.size() > maxPositionHistorySize) positionSmoothingHistory.pop_front();
        return positionSmoothingHistory;
    }()) : p0;

    // 2nd order prediction (acceleration)
    glm::vec3 v1 = (p1 - p2) / dt1;
    glm::vec3 v2 = (p0 - p1) / dt2;
    glm::vec3 v = 0.5f * (v1 + v2);
    glm::vec3 a = (v2 - v1) / dt2;
    a = glm::clamp(a, -3.0f, 3.0f);

    glm::vec3 rawPrediction = filteredP0 + v * dtFuture + 0.5f * a * dtFuture * dtFuture;

    float confidence = 1.0f - glm::smoothstep(0.02f, 0.06f, dtFuture);
    glm::vec3 finalPrediction = enableSmoothing ? glm::mix(filteredP0, rawPrediction, confidence) : rawPrediction;

    // Rotation prediction
    glm::quat dq = glm::normalize(r0 * glm::inverse(r1));
    float angle = glm::angle(dq);
    glm::vec3 axis = glm::axis(dq);
    if (glm::length(axis) < 1e-5f || glm::any(glm::isnan(axis))) axis = glm::vec3(0, 1, 0);

    // Angular velocity
    float angularSpeed = angle / dt2;
    angularSpeed = glm::clamp(angularSpeed, 0.0f, glm::radians(200.0f));

    float futureAngle = angularSpeed * dtFuture;
    futureAngle = glm::clamp(futureAngle, 0.0f, glm::radians(45.0f));

    glm::quat deltaFuture = glm::angleAxis(futureAngle, axis);
    glm::quat predictedRotation = glm::normalize(deltaFuture * r0);

    glm::quat finalRotation = enableSmoothing ? averageQuaternions([&] {
        rotationSmoothingHistory.push_back(predictedRotation);
        if (rotationSmoothingHistory.size() > maxRotationHistorySize) rotationSmoothingHistory.pop_front();
        return rotationSmoothingHistory;
    }()) : predictedRotation;

    glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), finalPrediction) * glm::mat4_cast(finalRotation);

    predictedPose.setViewMatrix(glm::inverse(predictedTransform));
    predictedPose.setProjectionMatrix(latest.mono.proj);

    return true;
}

void PosePredictor::accumulateError(const PerspectiveCamera& camera, const PerspectiveCamera& remoteCamera) {
    float positionDiff = glm::distance(camera.getPosition(), remoteCamera.getPosition());
    glm::quat q1 = glm::normalize(camera.getRotationQuat());
    glm::quat q2 = glm::normalize(remoteCamera.getRotationQuat());
    if (glm::dot(q1, q2) < 0.0f) q2 = -q2;

    float angleDiffRadians = 2.0f * glm::acos(glm::clamp(glm::dot(q1, q2), -1.0f, 1.0f));
    float angleDiffDegrees = glm::degrees(angleDiffRadians);

    positionErrors.push_back(positionDiff);
    rotationErrors.push_back(std::abs(angleDiffDegrees));
}

PosePredictor::ErrorStats PosePredictor::getErrorStats() const {
    ErrorStats stats;
    stats.positionErrMeanStd.x = calculateMean(positionErrors);
    stats.positionErrMeanStd.y = calculateStdDev(positionErrors, stats.positionErrMeanStd.x);
    stats.positionErrMinMax.x = positionErrors.empty() ? 0.0 : *std::min_element(positionErrors.begin(), positionErrors.end());
    stats.positionErrMinMax.y = positionErrors.empty() ? 0.0 : *std::max_element(positionErrors.begin(), positionErrors.end());

    stats.rotationErrMeanStd.x = calculateMean(rotationErrors);
    stats.rotationErrMeanStd.y = calculateStdDev(rotationErrors, stats.rotationErrMeanStd.x);
    stats.rotationErrMinMax.x = rotationErrors.empty() ? 0.0 : *std::min_element(rotationErrors.begin(), rotationErrors.end());
    stats.rotationErrMinMax.y = rotationErrors.empty() ? 0.0 : *std::max_element(rotationErrors.begin(), rotationErrors.end());

    return stats;
}

void PosePredictor::clearErrors() {
    positionErrors.clear();
    rotationErrors.clear();
}

double PosePredictor::calculateMean(const std::vector<double>& errors) const {
    if (errors.empty()) return 0.0;
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

double PosePredictor::calculateStdDev(const std::vector<double>& errors, double mean) const {
    if (errors.size() < 2) return 0.0;
    double sumSquaredDiffs = 0.0;
    for (double err : errors) {
        sumSquaredDiffs += (err - mean) * (err - mean);
    }
    return std::sqrt(sumSquaredDiffs / (errors.size() - 1));
}
