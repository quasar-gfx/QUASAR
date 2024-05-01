#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include <Lights/Light.h>
#include <CubeMap.h>
#include <Framebuffer.h>
#include <Materials/PointShadowMapMaterial.h>

struct PointLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec3 position = glm::vec3(0.0f);
    float intensity = 1.0f;
    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;
    float zNear = 0.1f;
    float zFar = 100.0f;
};

class PointLight : public Light {
public:
    glm::vec3 position = glm::vec3(0.0f);
    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;

    unsigned int channel = -1;

    glm::mat4 lookAtPerFace[NUM_CUBEMAP_FACES];
    PointLightShadowBuffer shadowMapFramebuffer;
    PointShadowMapMaterial shadowMapMaterial;

    explicit PointLight(const PointLightCreateParams &params)
            : position(params.position), constant(params.constant), linear(params.linear), quadratic(params.quadratic),
                Light(params.color, params.intensity, params.zNear, params.zFar) {
        shadowMapFramebuffer.createColorAndDepthBuffers(shadowRes, shadowRes);

        shadowProjectionMat = glm::perspective(glm::radians(90.0f), 1.0f, params.zNear, params.zFar);

        updateLookAtFace();
    }

    void setChannel(int channel) {
        this->channel = channel;
    }

    void bindMaterial(Material* material) override {
        std::string idxStr = std::to_string(this->channel);
        material->shader->setVec3("pointLights["+idxStr+"].position", position);
        material->shader->setVec3("pointLights["+idxStr+"].color", color);
        material->shader->setFloat("pointLights["+idxStr+"].intensity", intensity);
        material->shader->setFloat("pointLights["+idxStr+"].constant", constant);
        material->shader->setFloat("pointLights["+idxStr+"].linear", linear);
        material->shader->setFloat("pointLights["+idxStr+"].quadratic", quadratic);
        material->shader->setFloat("pointLights["+idxStr+"].farPlane", zFar);
    }

    void setPosition(const glm::vec3 &position) {
        this->position = position;
        updateLookAtFace();
    }

    void setAttenuation(float constant, float linear, float quadratic) {
        this->constant = constant;
        this->linear = linear;
        this->quadratic = quadratic;
    }

    static const unsigned int maxPointLights = 4;

private:
    void updateLookAtFace() {
        lookAtPerFace[0] = glm::lookAt(position, position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        lookAtPerFace[1] = glm::lookAt(position, position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        lookAtPerFace[2] = glm::lookAt(position, position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        lookAtPerFace[3] = glm::lookAt(position, position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
        lookAtPerFace[4] = glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        lookAtPerFace[5] = glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    }
};

#endif // POINT_LIGHT_H
