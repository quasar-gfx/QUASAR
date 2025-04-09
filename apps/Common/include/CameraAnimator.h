#ifndef ANIMATOR_H
#define ANIMATOR_H

#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <Cameras/PerspectiveCamera.h>

namespace quasar {

class CameraAnimator {
public:
    struct CameraPose {
        glm::vec3 position;
        glm::quat rotation;
        double timestamp;
    };

    bool running = false;

    double now = 0.0;
    double dt = 0.0;

    CameraAnimator(const std::string &pathFile, bool tween = true);

    void loadAnimation(const std::string &pathFile);
    bool update(double dt);

    void copyPoseToCamera(PerspectiveCamera &camera) const;

private:
    bool tween = false;

    std::vector<CameraPose> waypoints;
    size_t currentIndex = 0;

    const glm::vec3 getCurrentPosition() const;
    const glm::quat getCurrentRotation() const;
};

} // namespace quasar

#endif // ANIMATOR_H
