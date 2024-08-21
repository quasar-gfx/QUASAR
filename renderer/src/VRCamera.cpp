#include <VRCamera.h>

VRCamera::VRCamera()
    : left()
    , right() {}

VRCamera::VRCamera(unsigned int width, unsigned int height)
    : left(std::make_unique<PerspectiveCamera>(width, height))
    , right(std::make_unique<PerspectiveCamera>(width, height)) {}

VRCamera::VRCamera(float fovy, float aspect, float near, float far)
    : left(std::make_unique<PerspectiveCamera>(fovy, aspect, near, far))
    , right(std::make_unique<PerspectiveCamera>(fovy, aspect, near, far)) {}


bool VRCamera::isVR() const {
    return true;
}

void VRCamera::setFovy(float fovy) {
    left->setFovy(fovy);
    right->setFovy(fovy);
}

void VRCamera::setAspect(float aspect) {
    left->setAspect(aspect);
    right->setAspect(aspect);
}

void VRCamera::setNear(float near) {
    left->setNear(near);
    right->setNear(near);
}

void VRCamera::setFar(float far) {
    left->setFar(far);
    right->setFar(far);
}

void VRCamera::setProjectionMatrix(const glm::mat4 &proj) {
    left->setProjectionMatrix(proj);
    right->setProjectionMatrix(proj);
}

glm::mat4 VRCamera::getProjectionMatrix() const {
    return left->getProjectionMatrix();
}

glm::mat4 VRCamera::getEyeViewMatrix(bool isLeftEye) const {
    if (isLeftEye) {
        return left->getViewMatrix();
    } else {
        return right->getViewMatrix();
    }
}

void VRCamera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    left->setProjectionMatrix(fovy, aspect, near, far);
    right->setProjectionMatrix(fovy, aspect, near, far);
}

void VRCamera::setViewMatrices(const glm::mat4 (&views)[2]) {
    left->setViewMatrix(views[0]);
    right->setViewMatrix(views[1]);
}
