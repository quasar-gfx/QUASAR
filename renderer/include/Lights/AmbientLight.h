#ifndef AMBIENT_LIGHT_H
#define AMBIENT_LIGHT_H

#include <Lights/Light.h>

struct AmbientLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
};

class AmbientLight : public Light {
public:
    AmbientLight(const AmbientLightCreateParams &params)
        : Light({
            .color = params.color,
            .intensity = params.intensity
        }) {}

    void bindMaterial(const Material* material) override {
        material->shader->setVec3("ambientLight.color", color);
        material->shader->setFloat("ambientLight.intensity", intensity);
    }
};

#endif // AMBIENT_LIGHT_H
