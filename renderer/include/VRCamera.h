#ifndef VRCAMERA_H
#define VRCAMERA_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <Primatives/Node.h>
#include <Windowing/Window.h>
#include <Culling/Frustum.h>
#include <utility>
#include <Camera.h>

class VRCamera : public Node {
public:
    Camera left;
    Camera right;

    explicit VRCamera();
    VRCamera(Camera left, Camera right);
    void setViewMatrices(glm::mat4 views[]);

    glm::vec3 getPosition() const override {
        glm::vec3 leftPos = left.getPosition();
        glm::vec3 rightPos = right.getPosition();
        return (leftPos + rightPos) / 2.0f;
    }
    void setViewMatrix(glm::mat4 view[]);
};

#endif // VRCAMERA_H