#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <Lights/Light.h>
#include <RenderTargets/DirLightShadowRT.h>
#include <Materials/DirShadowMapMaterial.h>

namespace quasar {

struct DirectionalLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec3 direction = glm::vec3(0.0f, -5.0f, 1.333f);
    float distance = 100.0f;
    float intensity = 1.0f;
    float orthoBoxSize = 75.0f;
    float shadowNear = 1.0f;
    float shadowFar = 750.0f;
    uint shadowMapRes = 2048;
};

class DirectionalLight : public Light {
public:
    glm::vec3 direction = glm::vec3(0.0f);
    float distance = 100.0f;

    float orthoBoxSize = 75.0f;

    glm::mat4 lightView = glm::mat4(0.0);
    glm::mat4 lightSpaceMatrix = glm::mat4(0.0);

    DirLightShadowRT shadowMapRenderTarget;
    DirShadowMapMaterial shadowMapMaterial;

    DirectionalLight(const DirectionalLightCreateParams& params)
            : direction(params.direction)
            , distance(params.distance)
            , orthoBoxSize(params.orthoBoxSize)
            , Light({
                .color = params.color,
                .intensity = params.intensity,
                .shadowNear = params.shadowNear,
                .shadowFar = params.shadowFar,
                .shadowMapRes = params.shadowMapRes
            })
            , shadowMapRenderTarget({ .width = shadowMapRes, .height = shadowMapRes }) {
        updateLightSpaceMatrix();
    }

    void bindMaterial(const Material* material) override {
        material->getShader()->setVec3("directionalLight.direction", direction);
        material->getShader()->setVec3("directionalLight.color", color);
        material->getShader()->setFloat("directionalLight.intensity", intensity);
        updateLightSpaceMatrix();
    }

private:
    void updateLightSpaceMatrix() {
        float left = orthoBoxSize / 2;
        float right = -left;
        float top = left;
        float bottom = -top;
        shadowProjectionMat = glm::ortho(left, right, bottom, top, shadowNear, shadowFar);
        lightView = glm::lookAt(distance * -direction, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        lightSpaceMatrix = shadowProjectionMat * lightView;
    }
};

} // namespace quasar

#endif // DIRECTIONAL_LIGHT_H
