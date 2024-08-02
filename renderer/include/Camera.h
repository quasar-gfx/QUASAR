#ifndef CAMERA_H
#define CAMERA_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <Primatives/Node.h>
#include <Windowing/Window.h>
#include <Culling/Frustum.h>

class Camera : public Node {
public:
    float aspect;
    float fovy;
    float near;
    float far;

    float movementSpeed = 5.0f;
    float mouseSensitivity = 0.05f;

    Frustum frustum;

    explicit Camera(unsigned int width, unsigned int height);
    explicit Camera(float fovy, float aspect, float near, float far);

    void setFovy(float fovy) { this->fovy = fovy; updateProjectionMatrix(); }
    void setAspect(float aspect) { this->aspect = aspect; updateProjectionMatrix(); }
    void setNear(float near) { this->near = near; updateProjectionMatrix(); }
    void setFar(float far) { this->far = far; updateProjectionMatrix(); }

    glm::mat4 getProjectionMatrix() const { return proj; }
    void setProjectionMatrix(glm::mat4 proj);
    void setProjectionMatrix(float fovy, float aspect, float near, float far);
    void updateProjectionMatrix();

    glm::mat4 getViewMatrix() const { return view; }
    void setViewMatrix(glm::mat4 view);
    void updateViewMatrix();

    glm::vec3 getForwardVector() const { return front; }
    glm::vec3 getRightVector() const { return right; }
    glm::vec3 getUpVector() const { return up; }

    void processKeyboard(Keys keys, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

protected:
    glm::mat4 view;
    glm::mat4 proj;

    float yaw = -90.0f;
    float pitch = 0.0f;

    glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::vec3(-1.0f, 0.0f, 0.0f);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

    void updateCameraOrientation();
    // calculates the front vector from the Camera's (updated) Euler Angles
    void setOrientationFromYawPitch();
};

#endif // CAMERA_H
