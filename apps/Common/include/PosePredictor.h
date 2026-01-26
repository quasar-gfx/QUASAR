#ifndef POSE_PREDICTOR_H
#define POSE_PREDICTOR_H

#include <deque>
#include <vector>
#include <numeric>
#include <cmath>
#include <array>

#include <CameraPose.h>
#include <Cameras/PerspectiveCamera.h>

namespace quasar {

struct PosePredictorCreateParams {
    bool enablePrediction = false;
    bool enableSmoothing = false;
};

class PosePredictor {
public:
    bool enablePrediction;
    bool enableSmoothing;

    struct ErrorStats {
        glm::vec2 positionErrMeanStd;
        glm::vec2 positionErrMinMax;
        glm::vec2 rotationErrMeanStd;
        glm::vec2 rotationErrMinMax;
    };

    PosePredictor(PosePredictorCreateParams params);

    void addPose(const Pose& pose);
    bool predictPose(Pose& predictedPose, double targetFutureTimeS);
    bool predictPose(Pose& predictedPose, const Pose& latest, const Pose& previous, const Pose& secondPrevious, double targetFutureTimeS);

    void accumulateError(const PerspectiveCamera& camera, const PerspectiveCamera& remoteCamera);
    ErrorStats getErrorStats() const;
    void clearErrors();

private:
    std::deque<Pose> poseHistory;
    std::vector<double> positionErrors;
    std::vector<double> rotationErrors;

    std::deque<glm::vec3> positionSmoothingHistory;
    std::deque<glm::quat> rotationSmoothingHistory;
    static constexpr size_t maxPositionHistorySize = 10;
    static constexpr size_t maxRotationHistorySize = 10;

    glm::vec3 savitzkyGolayFilter(const std::deque<glm::vec3>& buffer);
    glm::quat averageQuaternions(const std::deque<glm::quat>& quats);

    double calculateMean(const std::vector<double>& errors) const;
    double calculateStdDev(const std::vector<double>& errors, double mean) const;
};

} // namespace quasar

#endif // POSE_PREDICTOR_H
