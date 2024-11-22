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

    virtual float getFovyRadians() const = 0;
    virtual float getFovyDegrees() const = 0;
    virtual float getAspect() const = 0;
    virtual float getNear() const = 0;
    virtual float getFar() const = 0;

    virtual bool isVR() const = 0;
};

#endif // CAMERA_H
