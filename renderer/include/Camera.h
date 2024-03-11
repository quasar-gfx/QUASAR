#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
public:
    glm::vec3 position = glm::vec3(0.0f);
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

    float movementSpeed = 2.5f;
    float mouseSensitivity = 0.1f;

    Camera(unsigned int width, unsigned int height) {
        setProjectionMatrix(glm::radians(60.0f), (float)width / (float)height, 0.1f, 1000.0f);
        position = glm::vec3(0.0f, 1.6f, 2.0f);
        updateViewMatrix();
    }

    Camera(float fovy, float aspect, float near, float far) {
        setProjectionMatrix(fovy, aspect, near, far);
        updateViewMatrix();
    }

    glm::mat4 getProjectionMatrix() {
        return proj;
    }

    void setProjectionMatrix(float fovy, float aspect, float near, float far) {
        this->fovy = fovy;
        this->aspect = aspect;
        this->near = near;
        this->far = far;
        updateProjectionMatrix();
    }

    void updateProjectionMatrix() {
        proj = glm::perspective(fovy, aspect, near, far);
    }

    glm::mat4 getViewMatrix() {
        return view;
    }

    void setViewMatrix(glm::mat4 view) {
        this->view = view;
    }

    void updateViewMatrix() {
        view = glm::lookAt(position, position + front, up);
    }

    void processKeyboard(CameraMovement direction, float deltaTime) {
        float velocity = movementSpeed * deltaTime;
        if (direction == FORWARD)
            position += front * velocity;
        if (direction == BACKWARD)
            position -= front * velocity;
        if (direction == LEFT)
            position -= right * velocity;
        if (direction == RIGHT)
            position += right * velocity;

        updateViewMatrix();
    }

    void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw   += xoffset;
        pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch) {
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;
        }

        updateCameraVectors();
        updateViewMatrix();
    }

    void processMouseScroll(float yoffset) {
        fovy -= (float)yoffset;

        if (fovy < glm::radians(1.0f)) {
            fovy = glm::radians(1.0f);
        }
        if (fovy > glm::radians(60.0f)) {
            fovy = glm::radians(60.0f);
        }
        updateProjectionMatrix();
    }

private:
    glm::mat4 view;
    glm::mat4 proj;

    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors() {
        // calculate the new front vector
        glm::vec3 newFront;
        newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        newFront.y = sin(glm::radians(pitch));
        newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(newFront);

        // also re-calculate the right and up vector
        right = glm::normalize(glm::cross(front, worldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        up    = glm::normalize(glm::cross(right, front));
    }
};

#endif // CAMERA_H
