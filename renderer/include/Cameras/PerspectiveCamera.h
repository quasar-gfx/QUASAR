#ifndef PERSPECTIVE_CAMERA_H
#define PERSPECTIVE_CAMERA_H

#include <Windowing/Window.h>
#include <Cameras/Camera.h>
#include <Culling/Frustum.h>

namespace quasar {

class PerspectiveCamera : public Camera {
public:
    float scrollSensitivity = 0.1f;
    float movementSpeed = 2.0f;
    float mouseSensitivity = 0.05f;

    Frustum frustum;

    PerspectiveCamera();
    PerspectiveCamera(uint width, uint height);
    PerspectiveCamera(float fovy, float aspect, float near, float far);

    float getFovyRadians() const override { return fovyRad; }
    float getFovyDegrees() const override { return glm::degrees(fovyRad); }
    void setFovyRadians(float fovyRad) { this->fovyRad = fovyRad; updateProjectionMatrix(); }
    void setFovyDegrees(float fovyDeg) { this->fovyRad = glm::radians(fovyDeg); updateProjectionMatrix(); }
    void setFovxDegrees(float fovxDeg) {
        float fovxRad = glm::radians(fovxDeg);
        float newFovyRad = 2.0f * glm::atan(glm::tan(fovxRad / 2.0f) / aspect);
        setFovyRadians(newFovyRad);
    }

    float getAspect() const override { return aspect; }
    void setAspect(float aspect) { this->aspect = aspect; updateProjectionMatrix(); }
    void setAspect(uint width, uint height) { setAspect((float)width / (float)height); }

    float getNear() const override { return near; }
    void setNear(float near) { this->near = near; updateProjectionMatrix(); }

    float getFar() const override { return far; }
    void setFar(float far) { this->far = far; updateProjectionMatrix(); }

    const glm::mat4& getProjectionMatrix() const { return proj; }
    const glm::mat4& getProjectionMatrixInverse() const { return projInverse; }
    void setProjectionMatrix(const glm::mat4& proj);
    void setProjectionMatrix(float fovyDeg, float aspect, float near, float far);
    void updateProjectionMatrix();

    const glm::mat4& getViewMatrix() const { return view; }
    const glm::mat4& getViewMatrixInverse() const { return viewInverse; }
    void setViewMatrix(const glm::mat4& view);
    void updateViewMatrix();

    glm::vec3 getForwardVector() const { return front; }
    glm::vec3 getRightVector() const { return right; }
    glm::vec3 getUpVector() const { return up; }

    void processKeyboard(Keys keys, double deltaTime);
    void processScroll(float yoffset);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

    bool isVR() const override { return false; }

private:
    float aspect;
    float fovyRad;
    float near;
    float far;

    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 projInverse;
    glm::mat4 viewInverse;

    float yaw = -90.0f;
    float pitch = 0.0f;

    glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::vec3(-1.0f, 0.0f, 0.0f);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

    void updateCameraOrientation();
    // Calculates the front vector from the PerspectiveCamera's (updated) Euler Angles
    void setOrientationFromYawPitch();
};

} // namespace quasar

#endif // PERSPECTIVE_CAMERA_H
