#ifndef TEXTURED_MATERIAL_H
#define TEXTURED_MATERIAL_H

#include <Materials/Material.h>

struct UnlitMaterialCreateParams {
    glm::vec3 color = glm::vec3(-1.0f);
    float opacity = 0.0f;
    bool transparent = false;
    float maskThreshold = 0.1f;
    std::string diffuseTexturePath = "";
    TextureID diffuseTextureID;
};

class UnlitMaterial : public Material {
public:
    glm::vec3 color = glm::vec3(-1.0f);
    float opacity = 0.0f;
    bool transparent = false;
    float maskThreshold = 0.1f;

    explicit UnlitMaterial() = default;
    explicit UnlitMaterial(const UnlitMaterialCreateParams &params);

    void bind() const override;

    unsigned int getTextureCount() const override { return 1; }
};

#endif // TEXTURED_MATERIAL_H
