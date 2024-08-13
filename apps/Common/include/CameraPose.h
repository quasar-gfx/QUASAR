#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <glm/glm.hpp>

typedef uint32_t pose_id_t;

struct Pose {
    pose_id_t id;
#ifdef VR
    glm::mat4 viewL;
    glm::mat4 viewR;
    glm::mat4 projL;
    glm::mat4 projR;    
#else
    glm::mat4 proj;
    glm::mat4 view;
#endif
    int timestamp;
};

#endif // CAMERA_POSE_H
