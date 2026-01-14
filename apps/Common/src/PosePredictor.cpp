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

bool PosePredictor::predictPose(Pose& predictedPose, double targetFutureTimeS) {
    if (poseHistory.size() < 2) {
        if (!poseHistory.empty()) {
            predictedPose = poseHistory.back();
        }
        return false;
    }

    return predictPose(predictedPose,
                      poseHistory[poseHistory.size()-1],
                      poseHistory[poseHistory.size()-2],
                      targetFutureTimeS);
}

bool PosePredictor::predictPose(
    Pose& predictedPose,
    const Pose& latest, const Pose& previous,
    double targetFutureTimeS)
{
    if (!enablePrediction) {
        predictedPose = latest;
        return true;
    }

    // Simple verification of timestamps
    double t1 = timeutils::microsToSeconds(previous.timestamp);
    double t0 = timeutils::microsToSeconds(latest.timestamp);
    float dt = static_cast<float>(t0 - t1);

    if (dt <= 1e-5f) return false;

    float dtFuture = static_cast<float>(targetFutureTimeS - t0);
    // Limit prediction time to avoid wild extrapolations
    dtFuture = glm::clamp(dtFuture, 0.0f, 0.1f);

    glm::vec3 scale, skew;
    glm::vec4 perspective;
    glm::vec3 p0, p1;
    glm::quat r0, r1;

    // Decompose the view matrices
    // View matrix is the inverse of the camera transform
    glm::decompose(glm::inverse(latest.mono.view), scale, r0, p0, skew, perspective);
    glm::decompose(glm::inverse(previous.mono.view), scale, r1, p1, skew, perspective);

    // Ensure quaternions are in the same neighborhood
    if (glm::dot(r0, r1) < 0.0f) r1 = -r1;

    // Linear Position Prediction
    // P_pred = P0 + V * t
    glm::vec3 velocity = (p0 - p1) / dt;
    glm::vec3 pPred = p0 + velocity * dtFuture;

    // Spherical Linear Integration for Rotation
    // Extrapolate rotation using Slerp
    // ratio = (time_total) / time_segment = (dt + dtFuture) / dt = 1 + dtFuture/dt
    float slerpRatio = 1.0f + dtFuture / dt;
    glm::quat rPred = glm::normalize(glm::slerp(r1, r0, slerpRatio));

    // Reconstruct the view matrix
    glm::mat4 predTransform = glm::translate(glm::mat4(1.0f), pPred) * glm::mat4_cast(rPred);

    predictedPose.setViewMatrix(glm::inverse(predTransform));
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
