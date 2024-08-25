#ifndef CAMERA_H
#define CAMERA_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Primatives/Node.h>

class Camera : public Node {
public:
    Camera() = default;
    virtual ~Camera() = default;

    virtual bool isVR() const = 0;
};

#endif // CAMERA_H
