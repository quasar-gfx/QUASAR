#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <Light.h>
#include <FrameBuffer.h>

class DirectionalLight : public Light {
public:
    glm::vec3 direction = glm::vec3(0.0f);

    float orthoBoxSize = 100.0f;

    glm::mat4 lightView = glm::mat4(0.0);
    glm::mat4 lightSpaceMatrix = glm::mat4(0.0);

    DirShadowBuffer dirLightShadowMapFBO;

    DirectionalLight(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f, float orthoBoxSize = 100.0f)
        : Light(color, intensity), orthoBoxSize(orthoBoxSize) {
        dirLightShadowMapFBO.createColorAndDepthBuffers(2048, 2048);
    }

    void draw(Shader &shader) {
        shader.setVec3("directionalLight.direction", direction);
        shader.setVec3("directionalLight.color", color);
        shader.setFloat("directionalLight.intensity", intensity);
    }

    void setDirection(const glm::vec3 &direction) {
        this->direction = direction;
    }
};

#endif // DIRECTIONAL_LIGHT_H
