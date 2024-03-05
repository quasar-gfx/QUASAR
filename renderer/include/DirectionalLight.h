#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <Light.h>

class DirectionalLight : public Light {
public:
    glm::vec3 direction = glm::vec3(0.0f);

    void draw(Shader &shader) {
        shader.setVec3("directionalLight.direction", direction);
        shader.setVec3("directionalLight.color", color);
        shader.setFloat("directionalLight.intensity", intensity);
    }

    void setDirection(const glm::vec3 &direction) {
        this->direction = direction;
    }

    static DirectionalLight* create(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) {
        return new DirectionalLight(color, intensity);
    }

protected:
    DirectionalLight(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) : Light(color, intensity) {}
};

#endif // DIRECTIONAL_LIGHT_H
