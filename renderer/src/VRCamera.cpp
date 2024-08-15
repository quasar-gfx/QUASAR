#include <VRCamera.h>

VRCamera::VRCamera()
    : left()
    , right() {}

VRCamera::VRCamera(unsigned int width, unsigned int height)
    : left(width, height)
    , right(width, height) {}

VRCamera::VRCamera(float fovy, float aspect, float near, float far)
    : left(fovy, aspect, near, far)
    , right(fovy, aspect, near, far) {}

void VRCamera::setFovy(float fovy) {
    left.setFovy(fovy);
    right.setFovy(fovy);
}

void VRCamera::setAspect(float aspect) {
    left.setAspect(aspect);
    right.setAspect(aspect);
}

void VRCamera::setNear(float near) {
    left.setNear(near);
    right.setNear(near);
}

void VRCamera::setFar(float far) {
    left.setFar(far);
    right.setFar(far);
}

void VRCamera::setProjectionMatrix(const glm::mat4 &proj) {
    left.setProjectionMatrix(proj);
    right.setProjectionMatrix(proj);
}

void VRCamera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    left.setProjectionMatrix(fovy, aspect, near, far);
    right.setProjectionMatrix(fovy, aspect, near, far);
}

void VRCamera::setViewMatrices(const glm::mat4 (&views)[2]) {
    left.setViewMatrix(views[0]);
    right.setViewMatrix(views[1]);
}
