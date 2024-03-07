#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include <Light.h>
#include <CubeMap.h>
#include <FrameBuffer.h>

class PointLight : public Light {
public:
    glm::vec3 position = glm::vec3(0.0f);
    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;

    glm::mat4 lookAtPerFace[NUM_CUBEMAP_FACES];
    PointShadowBuffer pointLightShadowMapFBO;

    PointLight(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f, float zNear = 1.0f, float zFar = 25.0f)
        : Light(color, intensity) {
        pointLightShadowMapFBO.createColorAndDepthBuffers(shadowRes, shadowRes);

        shadowProjectionMat = glm::perspective(glm::radians(90.0f), 1.0f, zNear, zFar);

        updateLookAtFace();
    }

    void draw(Shader &shader) {
        draw(shader, 0);
    }

    void draw(Shader &shader, int idx) {
        std::string idxStr = std::to_string(idx);
        shader.setVec3("pointLights["+idxStr+"].position", position);
        shader.setVec3("pointLights["+idxStr+"].color", color);
        shader.setFloat("pointLights["+idxStr+"].intensity", intensity);
        shader.setFloat("pointLights["+idxStr+"].constant", constant);
        shader.setFloat("pointLights["+idxStr+"].linear", linear);
        shader.setFloat("pointLights["+idxStr+"].quadratic", quadratic);
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
