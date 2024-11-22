#include <Cameras/VRCamera.h>

VRCamera::VRCamera()
    : left()
    , right() {
    left.setPosition(glm::vec3(-0.032f, 0.0f, 0.0f));
    right.setPosition(glm::vec3(0.032f, 0.0f, 0.0f));

    addChildNode(&left);
    addChildNode(&right);
}

VRCamera::VRCamera(unsigned int width, unsigned int height)
    : left(PerspectiveCamera(width, height))
    , right(PerspectiveCamera(width, height)) {
    left.setPosition(glm::vec3(-0.032f, 0.0f, 0.0f));
    right.setPosition(glm::vec3(0.032f, 0.0f, 0.0f));

    addChildNode(&left);
    addChildNode(&right);
}

VRCamera::VRCamera(float fovy, float aspect, float near, float far)
    : left(PerspectiveCamera(fovy, aspect, near, far))
    , right(PerspectiveCamera(fovy, aspect, near, far)) {
    left.setPosition(glm::vec3(-0.032f, 0.0f, 0.0f));
    right.setPosition(glm::vec3(0.032f, 0.0f, 0.0f));
}

void VRCamera::setProjectionMatrix(const glm::mat4 &proj) {
    left.setProjectionMatrix(proj);
    right.setProjectionMatrix(proj);
}

glm::mat4 VRCamera::getProjectionMatrix() const {
    return left.getProjectionMatrix();
}

void VRCamera::updateProjectionMatrix() {
    left.updateProjectionMatrix();
    right.updateProjectionMatrix();
}

glm::mat4 VRCamera::getEyeViewMatrix(bool isLeftEye) const {
    if (isLeftEye) {
        return left.getViewMatrix();
    }
    else {
        return right.getViewMatrix();
    }
}

void VRCamera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    left.setProjectionMatrix(fovy, aspect, near, far);
    right.setProjectionMatrix(fovy, aspect, near, far);
}

void VRCamera::setProjectionMatrices(const glm::mat4 (&projs)[2]) {
    left.setProjectionMatrix(projs[0]);
    right.setProjectionMatrix(projs[1]);
}

void VRCamera::setViewMatrices(const glm::mat4 (&views)[2]) {
    left.setViewMatrix(views[0]);
    right.setViewMatrix(views[1]);
}
