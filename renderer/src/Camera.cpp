#include <Camera.h>

Camera::Camera(unsigned int width, unsigned int height) {
    setProjectionMatrix(glm::radians(60.0f), (float)width / (float)height, 0.1f, 1000.0f);
    updateCameraVectors();
}

Camera::Camera(float fovy, float aspect, float near, float far) {
    setProjectionMatrix(fovy, aspect, near, far);
    updateCameraVectors();
}

void Camera::setProjectionMatrix(glm::mat4 proj) {
    this->proj = proj;

    fovy = atan(1.0f / proj[1][1]) * 2.0f;
    aspect = proj[1][1] / proj[0][0];
    near = proj[3][2] / (proj[2][2] - 1.0f);
    far = proj[3][2] / (proj[2][2] + 1.0f);
}

void Camera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    this->fovy = fovy;
    this->aspect = aspect;
    this->near = near;
    this->far = far;
    updateProjectionMatrix();
}

void Camera::updateProjectionMatrix() {
    proj = glm::perspective(fovy, aspect, near, far);
    frustum.setFromCameraParams(position, front, right, up, near, far, aspect, fovy);
}

void Camera::setViewMatrix(glm::mat4 view) {
    this->view = view;

    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(glm::inverse(view), scale, rotation, position, skew, perspective);
    updateCameraVectors();
}

void Camera::updateViewMatrix() {
    view = glm::scale(glm::mat4(1.0f), 1.0f/scale) * glm::mat4_cast(glm::conjugate(rotation)) * glm::translate(glm::mat4(1.0f), -position);
    frustum.setFromCameraParams(position, front, right, up, near, far, aspect, fovy);
}

void Camera::processKeyboard(Keys keys, float deltaTime) {
    float velocity = movementSpeed * deltaTime;
    if (keys.W_PRESSED)
        position += front * velocity;
    if (keys.A_PRESSED)
        position -= right * velocity;
    if (keys.S_PRESSED)
        position -= front * velocity;
    if (keys.D_PRESSED)
        position += right * velocity;
    if (keys.Q_PRESSED)
        position += up * velocity;
    if (keys.E_PRESSED)
        position -= up * velocity;

    updateViewMatrix();
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
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
}

void Camera::updateCameraVectors() {
    // calculate the new front vector
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);

    // also re-calculate the right and up vector
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));

    glm::mat4 newView = glm::lookAt(position, position + front, up);

    glm::vec3 temp, skew;
    glm::vec4 perspective;
    glm::decompose(glm::inverse(newView), scale, rotation, temp, skew, perspective);

    updateViewMatrix();
}
