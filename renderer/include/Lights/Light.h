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
    float shadowFar = 25.0f;
    unsigned int shadowMapRes = 2048;
};

class Light {
public:
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;

    float shadowNear = 1.0f;
    float shadowFar = 2000.0f;

    glm::mat4 shadowProjectionMat = glm::mat4(0.0);

    unsigned int shadowMapRes = 2048;

    Light(const LightCreateParams &params)
            : color(params.color)
            , intensity(params.intensity)
            , shadowNear(params.shadowNear)
            , shadowFar(params.shadowFar)
            , shadowMapRes(params.shadowMapRes) { }

    virtual void bindMaterial(const Material* material) = 0;
};

} // namespace quasar

#endif // LIGHT_H
