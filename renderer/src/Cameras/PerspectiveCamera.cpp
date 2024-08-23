#include <Cameras/PerspectiveCamera.h>

PerspectiveCamera::PerspectiveCamera() {
    setProjectionMatrix(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
    updateCameraOrientation();
}

PerspectiveCamera::PerspectiveCamera(unsigned int width, unsigned int height) {
    setProjectionMatrix(glm::radians(60.0f), (float)width / (float)height, 0.1f, 1000.0f);
    updateCameraOrientation();
}

PerspectiveCamera::PerspectiveCamera(float fovy, float aspect, float near, float far) {
    setProjectionMatrix(fovy, aspect, near, far);
    updateCameraOrientation();
}

void PerspectiveCamera::setProjectionMatrix(const glm::mat4 &proj) {
    this->proj = proj;

    fovy = atan(1.0f / proj[1][1]) * 2.0f;
    aspect = proj[1][1] / proj[0][0];
    near = proj[3][2] / (proj[2][2] - 1.0f);
    far = proj[3][2] / (proj[2][2] + 1.0f);
}

void PerspectiveCamera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    this->fovy = fovy;
    this->aspect = aspect;
    this->near = near;
    this->far = far;
    updateProjectionMatrix();
}

void PerspectiveCamera::updateProjectionMatrix() {
    proj = glm::perspective(fovy, aspect, near, far);
    // frustum.setFromCameraParams(position, front, right, up, near, far, aspect, fovy);
    frustum.setFromCameraMatrices(view, proj);
}

void PerspectiveCamera::setViewMatrix(const glm::mat4 &view) {
    this->view = view;
    updateCameraOrientation();
    // frustum.setFromCameraParams(position, front, right, up, near, far, aspect, fovy);
    frustum.setFromCameraMatrices(view, proj);
}

void PerspectiveCamera::updateViewMatrix() {
    view = getTransformLocalFromParent();
    updateCameraOrientation();
    // frustum.setFromCameraParams(position, front, right, up, near, far, aspect, fovy);
    frustum.setFromCameraMatrices(view, proj);
}

void PerspectiveCamera::processKeyboard(Keys keys, float deltaTime) {
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
        position -= up * velocity;
    if (keys.E_PRESSED)
        position += up * velocity;

    updateViewMatrix();
}

void PerspectiveCamera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
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

    setOrientationFromYawPitch();
}

void PerspectiveCamera::updateCameraOrientation() {
    front = -glm::normalize(glm::vec3(view[0][2], view[1][2], view[2][2]));
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));

    yaw = glm::degrees(atan2(front.z, front.x));
    pitch = glm::degrees(asin(front.y));
}

void PerspectiveCamera::setOrientationFromYawPitch() {
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
    setTransformParentFromLocal(glm::inverse(newView));

    updateViewMatrix();
}
