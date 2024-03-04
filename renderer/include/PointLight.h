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
        shader.setVec3("pointLight.position", position);
        shader.setVec3("pointLight.color", color);
        shader.setFloat("pointLight.intensity", intensity);
        shader.setFloat("pointLight.constant", constant);
        shader.setFloat("pointLight.linear", linear);
        shader.setFloat("pointLight.quadratic", quadratic);
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

private:
    PointLight(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) : Light(color, intensity) {}
};

#endif // POINT_LIGHT_H
