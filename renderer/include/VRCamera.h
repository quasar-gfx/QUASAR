#ifndef VRCAMERA_H
#define VRCAMERA_H

#include <PerspectiveCamera.h>
#include <Culling/Frustum.h>
#include <Camera.h>

class VRCamera : public Camera {
public:
    std::unique_ptr<PerspectiveCamera> left;
    std::unique_ptr<PerspectiveCamera> right;

    explicit VRCamera();
    explicit VRCamera(unsigned int width, unsigned int height);
    explicit VRCamera(float fovy, float aspect, float near, float far);

    bool isVR() const override;
    void setFovy(float fovy);
    void setAspect(float aspect);
    void setNear(float near);
    void setFar(float far);

    void setProjectionMatrix(const glm::mat4 &proj);
    void setProjectionMatrix(float fovy, float aspect, float near, float far);

    void setViewMatrices(const glm::mat4 (&views)[2]);

    glm::vec3 getPosition() const override {
        // use the average of the two camera positions for the head center
        glm::vec3 leftPos = left->getPosition();
        glm::vec3 rightPos = right->getPosition();
        return (leftPos + rightPos) / 2.0f;
    }

    glm::mat4 getProjectionMatrix() const override;
    glm::mat4 getEyeViewMatrix(bool isLeftEye) const override; 
};

#endif // VRCAMERA_H