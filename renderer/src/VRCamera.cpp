#include <VRCamera.h>

VRCamera::VRCamera()
    : left()
    , right() {}

VRCamera::VRCamera(unsigned int width, unsigned int height)
    : left(PerspectiveCamera(width, height))
    , right(PerspectiveCamera(width, height)) {}

VRCamera::VRCamera(float fovy, float aspect, float near, float far)
    : left(PerspectiveCamera(fovy, aspect, near, far))
    , right(PerspectiveCamera(fovy, aspect, near, far)) {}

void VRCamera::setProjectionMatrix(const glm::mat4 &proj) {
    left.setProjectionMatrix(proj);
    right.setProjectionMatrix(proj);
}

glm::mat4 VRCamera::getProjectionMatrix() const {
    return left.getProjectionMatrix();
}

glm::mat4 VRCamera::getEyeViewMatrix(bool isLeftEye) const {
    if (isLeftEye) {
        return left.getViewMatrix();
    } else {
        return right.getViewMatrix();
    }
}

void VRCamera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    left.setProjectionMatrix(fovy, aspect, near, far);
    right.setProjectionMatrix(fovy, aspect, near, far);
}

void VRCamera::setViewMatrices(const glm::mat4 (&views)[2]) {
    left.setViewMatrix(views[0]);
    right.setViewMatrix(views[1]);
}
