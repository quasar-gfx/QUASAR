#include <VRCamera.h>

VRCamera::VRCamera() : left(), right() {}

VRCamera::VRCamera(Camera left, Camera right) {
    this->left = left;
    this->right = right;
}

void VRCamera::setViewMatrices(glm::mat4 views[]) {
    left.setViewMatrix(views[0]);
    right.setViewMatrix(views[1]);
}

void VRCamera::setViewMatrix(glm::mat4 view[]) {
    left.setViewMatrix(view[0]);
    right.setViewMatrix(view[0]);
}
