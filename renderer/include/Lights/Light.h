#ifndef LIGHT_H
#define LIGHT_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Primatives/Entity.h>
#include <Materials/Material.h>

struct LightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
    float zNear = 1.0f;
    float zFar = 25.0f;
    unsigned int shadowMapRes = 2048;
};

class Light {
public:
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;

    float zNear = 1.0f;
    float zFar = 2000.0f;

    glm::mat4 shadowProjectionMat = glm::mat4(0.0);

    unsigned int shadowMapRes = 2048;

    explicit Light(const LightCreateParams &params)
            : color(params.color)
            , intensity(params.intensity)
            , zNear(params.zNear)
            , zFar(params.zFar)
            , shadowMapRes(params.shadowMapRes) { }

    virtual void bindMaterial(const Material* material) = 0;
};

#endif // LIGHT_H
