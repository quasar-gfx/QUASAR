#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <glm/glm.hpp>

namespace quasar {

typedef uint32_t pose_id_t;

struct Pose {
    pose_id_t id;
    union {
        struct {
            glm::mat4 viewL;
            glm::mat4 viewR;
            glm::mat4 projL;
            glm::mat4 projR;
        } stereo;
        struct {
            glm::mat4 view;
            glm::mat4 pad1;
            glm::mat4 proj;
            glm::mat4 pad2;
        } mono;
    };
    uint64_t timestamp;

    Pose() = default;
    Pose(const glm::mat4 &view, const glm::mat4 &proj, uint64_t timestamp)
            : mono{view, glm::mat4(1.0f), proj, glm::mat4(1.0f)}, timestamp(timestamp) {}

    void setViewMatrix(const glm::mat4 &view) {
        mono.view = view;
    }

    void setProjectionMatrix(const glm::mat4 &proj) {
        mono.proj = proj;
    }

    void setViewMatrices(const glm::mat4 (&views)[2]) {
        stereo.viewL = views[0];
        stereo.viewR = views[1];
    }

    void setProjectionMatrices(const glm::mat4 (&projs)[2]) {
        stereo.projL = projs[0];
        stereo.projR = projs[1];
    }
};

} // namespace quasar

#endif // CAMERA_POSE_H
