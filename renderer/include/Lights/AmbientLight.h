#ifndef AMBIENT_LIGHT_H
#define AMBIENT_LIGHT_H

#include <Lights/Light.h>

namespace quasar {

struct AmbientLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
};

class AmbientLight : public Light {
public:
    struct GPUAmbientLight {
        glm::vec3 color;
        float intensity;
    };

    AmbientLight(const AmbientLightCreateParams& params)
        : Light({
            .color = params.color,
            .intensity = params.intensity,
        })
    {}

    void bindMaterial(const Material* material) override {
        const GPUAmbientLight gpuAmbientLight = toGPULight();
        material->getShader()->setVec3("ambientLight.color", gpuAmbientLight.color);
        material->getShader()->setFloat("ambientLight.intensity", gpuAmbientLight.intensity);
    }

    const GPUAmbientLight toGPULight() const {
        return {
            .color = color,
            .intensity = intensity,
        };
    }
};

} // namespace quasar

#endif // AMBIENT_LIGHT_H
