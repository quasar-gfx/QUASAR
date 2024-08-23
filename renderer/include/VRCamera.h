#ifndef VRCAMERA_H
#define VRCAMERA_H

#include <PerspectiveCamera.h>
#include <Culling/Frustum.h>
#include <Camera.h>

class VRCamera : public Camera {
public:
    PerspectiveCamera left;
    PerspectiveCamera right;

    VRCamera();
    VRCamera(unsigned int width, unsigned int height);
    VRCamera(float fovy, float aspect, float near, float far);

    void setProjectionMatrix(const glm::mat4 &proj);
    void setProjectionMatrix(float fovy, float aspect, float near, float far);

    void setViewMatrices(const glm::mat4 (&views)[2]);

    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getEyeViewMatrix(bool isLeftEye) const;

    glm::vec3 getPosition() const override {
        glm::vec3 leftPos = left.getPosition();
        glm::vec3 rightPos = right.getPosition();
        return (leftPos + rightPos) / 2.0f;
    }

    bool isVR() const override { return true; }
};

#endif // VRCAMERA_H