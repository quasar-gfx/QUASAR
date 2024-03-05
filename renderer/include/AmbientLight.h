#ifndef AMBIENT_LIGHT_H
#define AMBIENT_LIGHT_H

#include <Light.h>

class AmbientLight : public Light {
public:
    void draw(Shader &shader) {
        shader.setVec3("ambientLight.color", color);
        shader.setFloat("ambientLight.intensity", intensity);
    }

    static AmbientLight* create(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) {
        return new AmbientLight(color, intensity);
    }

protected:
    AmbientLight(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) : Light(color, intensity) {}
};

#endif // AMBIENT_LIGHT_H
