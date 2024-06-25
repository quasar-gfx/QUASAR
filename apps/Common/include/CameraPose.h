#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <glm/glm.hpp>

typedef uint32_t pose_id_t;

struct Pose {
    pose_id_t id;
    glm::mat4 proj;
    glm::mat4 view;
    int timestamp;
};

#endif // CAMERA_POSE_H
