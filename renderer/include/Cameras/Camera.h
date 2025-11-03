#ifndef CAMERA_H
#define CAMERA_H

#include <Primitives/Node.h>

#define DEFAULT_FOV_DEG 80.0f
#define DEFAULT_ASPECT 16.0f / 9.0f
#define DEFAULT_NEAR 0.05f
#define DEFAULT_FAR 1000.0f

namespace quasar {

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

} // namespace quasar

#endif // CAMERA_H
