#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include <Light.h>

class PointLight : public Light {
public:
    glm::vec3 position = glm::vec3(0.0f);
    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;

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
    }

    void setAttenuation(float constant, float linear, float quadratic) {
        this->constant = constant;
        this->linear = linear;
        this->quadratic = quadratic;
    }

    static PointLight* create(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) {
        return new PointLight(color, intensity);
    }

protected:
    PointLight(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) : Light(color, intensity) {}
};

#endif // POINT_LIGHT_H
