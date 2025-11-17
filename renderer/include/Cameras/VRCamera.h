#ifndef VR_CAMERA_H
#define VR_CAMERA_H

#include <Cameras/Camera.h>
#include <Cameras/PerspectiveCamera.h>
#include <Culling/Frustum.h>

#define DEFAULT_IPD 0.064f

namespace quasar {

class VRCamera : public Camera {
public:
    PerspectiveCamera left;
    PerspectiveCamera right;

    VRCamera();
    VRCamera(float fovyDeg, float aspect, float near, float far);
    VRCamera(const glm::uvec2& windowSize);
    VRCamera(uint width, uint height);
    VRCamera(const glm::mat4 (&projs)[2]);

    float getFovyRadians() const override { return left.getFovyRadians(); }
    float getFovyDegrees() const override { return left.getFovyDegrees(); }
    float getAspect() const override { return left.getAspect(); }
    float getFar() const override { return left.getFar(); }
    float getNear() const override { return left.getNear(); }

    void setFovyRadians(float fovyRad) override {
        left.setFovyRadians(fovyRad);
        right.setFovyRadians(fovyRad);
    }
    void setFovyDegrees(float fovyDeg) override {
        left.setFovyDegrees(fovyDeg);
        right.setFovyDegrees(fovyDeg);
    }
    void setAspect(float aspect) override {
        left.setAspect(aspect);
        right.setAspect(aspect);
    }
    void setNear(float near) override {
        left.setNear(near);
        right.setNear(near);
    }
    void setFar(float far) override {
        left.setFar(far);
        right.setFar(far);
    }

    void setProjectionMatrix(const glm::mat4& proj);
    void setProjectionMatrix(float fovy, float aspect, float near, float far);
    void setProjectionMatrices(const glm::mat4 (&projs)[2]);
    glm::mat4 getProjectionMatrix() const;
    void updateProjectionMatrix();

    void setViewMatrices(const glm::mat4 (&views)[2]);
    glm::mat4 getEyeViewMatrix(bool isLeftEye) const;

    glm::vec3 getPosition() const override {
        glm::vec3 leftPos = left.getPosition();
        glm::vec3 rightPos = right.getPosition();
        return (leftPos + rightPos) / 2.0f;
    }

    bool isVR() const override { return true; }
};

} // namespace quasar

#endif // VR_CAMERA_H
