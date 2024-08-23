#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Primatives/Node.h>


class Camera : public Node {
public:
    virtual bool isVR() const = 0;
    virtual ~Camera() = default;
};

#endif // CAMERA_H