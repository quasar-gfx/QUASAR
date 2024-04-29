#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <Windowing/Window.h>

class Camera {
public:
    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::vec3(-1.0f, 0.0f, 0.0f);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

    float aspect;
    float fovy;
    float near;
    float far;

    float yaw = -90.0f;
    float pitch = 0.0f;

    float movementSpeed = 5.0f;
    float mouseSensitivity = 0.05f;

    explicit Camera(unsigned int width, unsigned int height);
    explicit Camera(float fovy, float aspect, float near, float far);

    glm::mat4 getProjectionMatrix() { return proj; }
    void setProjectionMatrix(float fovy, float aspect, float near, float far);
    void updateProjectionMatrix();

    glm::mat4 getViewMatrix() { return view; }
    void setViewMatrix(glm::mat4 view);
    void updateViewMatrix();

    void processKeyboard(Keys keys, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

private:
    glm::mat4 view;
    glm::mat4 proj;

    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors();
};

#endif // CAMERA_H
