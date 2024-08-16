#ifndef VRCAMERA_H
#define VRCAMERA_H

#include <Camera.h>
#include <Culling/Frustum.h>

class VRCamera : public Node {
public:
    Camera left;
    Camera right;

    VRCamera();
    VRCamera(unsigned int width, unsigned int height);
    VRCamera(float fovy, float aspect, float near, float far);

    void setFovy(float fovy);
    void setAspect(float aspect);
    void setNear(float near);
    void setFar(float far);

    void setProjectionMatrix(const glm::mat4 &proj);
    void setProjectionMatrix(float fovy, float aspect, float near, float far);

    void setViewMatrices(const glm::mat4 (&views)[2]);

    glm::vec3 getPosition() const override {
        // use the average of the two camera positions for the head center
        glm::vec3 leftPos = left.getPosition();
        glm::vec3 rightPos = right.getPosition();
        return (leftPos + rightPos) / 2.0f;
    }
};

#endif // VRCAMERA_H
