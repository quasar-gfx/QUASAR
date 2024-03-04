#ifndef LIGHT_H
#define LIGHT_H

#include <Entity.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Light : public Entity {
public:
    Light(const glm::vec3 &color = glm::vec3(1.0f), float intensity = 1.0f) : color(color), intensity(intensity) {}

    void draw(Shader &shader) override = 0;

    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;

    EntityType getType() override { return ENTITY_LIGHT; }
};

#endif // LIGHT_H
