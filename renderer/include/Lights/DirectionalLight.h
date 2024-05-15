#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <memory>

#include <Lights/Light.h>
#include <RenderTargets/DirLightShadowRT.h>
#include <Materials/DirShadowMapMaterial.h>

struct DirectionalLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec3 direction = glm::vec3(0.0f, -5.0f, 1.333f);
    float distance = 100.0f;
    float intensity = 1.0f;
    float orthoBoxSize = 75.0f;
    float zNear = 1.0f;
    float zFar = 750.0f;
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

    explicit DirectionalLight(const DirectionalLightCreateParams &params)
            : direction(params.direction), distance(params.distance),
              orthoBoxSize(params.orthoBoxSize),
              Light(params.color, params.intensity, params.zNear, params.zFar) {
        shadowMapRenderTarget.init({
            .width = shadowRes,
            .height = shadowRes
        });
        updateLightSpaceMatrix();
    }

    void bindMaterial(Material* material) override {
        material->shader->setVec3("directionalLight.direction", direction);
        material->shader->setVec3("directionalLight.color", color);
        material->shader->setFloat("directionalLight.intensity", intensity);
        updateLightSpaceMatrix();
    }

private:
    void updateLightSpaceMatrix() {
        float left = orthoBoxSize / 2;
        float right = -left;
        float top = left;
        float bottom = -top;
        shadowProjectionMat = glm::ortho(left, right, bottom, top, zNear, zFar);
        lightView = glm::lookAt(distance * -direction, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        lightSpaceMatrix = shadowProjectionMat * lightView;
    }
};

#endif // DIRECTIONAL_LIGHT_H
