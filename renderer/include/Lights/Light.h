#ifndef LIGHT_H
#define LIGHT_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Primatives/Entity.h>
#include <Materials/Material.h>

class Light {
public:
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;

    float zNear = 1.0f;
    float zFar = 25.0f;

    glm::mat4 shadowProjectionMat = glm::mat4(0.0);

    unsigned int shadowRes = 2048;

    explicit Light(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f, float zNear = 1.0f, float zFar = 25.0f)
        : color(color), intensity(intensity), zNear(zNear), zFar(zFar) { }

    virtual void bindMaterial(Material* material) = 0;
};

#endif // LIGHT_H
