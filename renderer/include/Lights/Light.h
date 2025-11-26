#ifndef LIGHT_H
#define LIGHT_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Primitives/Entity.h>
#include <Materials/Material.h>

namespace quasar {

struct LightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
    float shadowNear = 1.0f;
    float shadowFar = 1000.0f;
    uint shadowMapRes = 1024;
};

class Light {
public:
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;

    float shadowNear;
    float shadowFar;

    uint shadowMapRes;

    glm::mat4 shadowProjectionMat{1.0f};

    Light(const LightCreateParams& params)
        : color(params.color)
        , intensity(params.intensity)
        , shadowNear(params.shadowNear)
        , shadowFar(params.shadowFar)
        , shadowMapRes(params.shadowMapRes)
    {}

    virtual void bindMaterial(const Material* material) = 0;
};

} // namespace quasar

#endif // LIGHT_H
