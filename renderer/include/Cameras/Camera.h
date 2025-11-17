#ifndef CAMERA_H
#define CAMERA_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Primitives/Node.h>

namespace quasar {

#define DEFAULT_FOV_DEG 80.0f
#define DEFAULT_ASPECT 16.0f / 9.0f
#define DEFAULT_NEAR 0.05f
#define DEFAULT_FAR 1000.0f

class Camera : public Node {
public:
    float movementSpeed = 2.0f;

    Camera() : Node("Camera" + std::to_string(nextID)) {}
    virtual ~Camera() = default;

    virtual float getFovyRadians() const = 0;
    virtual float getFovyDegrees() const = 0;
    virtual float getAspect() const = 0;
    virtual float getNear() const = 0;
    virtual float getFar() const = 0;

    virtual void setFovyRadians(float fovyRad) = 0;
    virtual void setFovyDegrees(float fovyDeg) = 0;
    virtual void setAspect(float aspect) = 0;
    virtual void setNear(float near) = 0;
    virtual void setFar(float far) = 0;

    virtual bool isVR() const = 0;
};

} // namespace quasar

#endif // CAMERA_H
