#ifndef AMBIENT_LIGHT_H
#define AMBIENT_LIGHT_H

#include <Light.h>

struct AmbientLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
};

class AmbientLight : public Light {
public:
    explicit AmbientLight(const AmbientLightCreateParams &params) : Light(params.color, params.intensity) {}

    void draw(Shader &shader) {
        shader.setVec3("ambientLight.color", color);
        shader.setFloat("ambientLight.intensity", intensity);
    }
};

#endif // AMBIENT_LIGHT_H
