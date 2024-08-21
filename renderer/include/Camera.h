#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Node.h>

class Camera : public Node {
protected:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float yaw;
    float pitch;

public:
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = -90.0f, float pitch = 0.0f);

    virtual ~Camera() = default;
    virtual bool isVR() const = 0;
    virtual glm::mat4 getViewMatrix() const;
    virtual glm::mat4 getProjectionMatrix() const = 0;
    virtual glm::mat4 getEyeViewMatrix(bool isLeftEye) const {
        // Default implementation just returns the regular view matrix
        return getViewMatrix();
    }
    void setPosition(const glm::vec3& newPosition);
    glm::vec3 getPosition() const;

    void setFront(const glm::vec3& newFront);
    glm::vec3 getFront() const;

    void setUp(const glm::vec3& newUp);
    glm::vec3 getUp() const;

    void setRight(const glm::vec3& newRight);
    glm::vec3 getRight() const;

    void setYaw(float newYaw);
    float getYaw() const;

    void setPitch(float newPitch);
    float getPitch() const;

protected:
    virtual void updateCameraVectors();
};
