#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <Lights/Light.h>
#include <Framebuffer.h>
#include <Materials/DirShadowMapMaterial.h>

struct DirectionalLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0);
    float intensity = 1.0f;
    float orthoBoxSize = 10.0f;
    float zNear = 1.0f;
    float zFar = 100.0f;
};

class DirectionalLight : public Light {
public:
    glm::vec3 direction = glm::vec3(0.0f);

    float orthoBoxSize = 10.0f;

    glm::mat4 lightView = glm::mat4(0.0);
    glm::mat4 lightSpaceMatrix = glm::mat4(0.0);

    DirLightShadowBuffer shadowMapFramebuffer;
    DirShadowMapMaterial shadowMapMaterial;

    explicit DirectionalLight(const DirectionalLightCreateParams &params)
            : direction(params.direction), orthoBoxSize(params.orthoBoxSize), Light(params.color, params.intensity, params.zNear, params.zFar) {
        shadowMapFramebuffer.createColorAndDepthBuffers(2048, 2048);
        updateLightView();
    }

    void bindMaterial(Material &material) override {
        material.shader->setVec3("directionalLight.direction", direction);
        material.shader->setVec3("directionalLight.color", color);
        material.shader->setFloat("directionalLight.intensity", intensity);
    }

    void setDirection(const glm::vec3 &direction) {
        this->direction = direction;
        updateLightView();
    }

private:
    void updateLightView() {
        float left = orthoBoxSize;
        float right = -left;
        float top = left;
        float bottom = -top;
        shadowProjectionMat = glm::ortho(left, right, bottom, top, zNear, zFar);
        lightView = glm::lookAt(-direction, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        lightSpaceMatrix = shadowProjectionMat * lightView;
    }
};

#endif // DIRECTIONAL_LIGHT_H
