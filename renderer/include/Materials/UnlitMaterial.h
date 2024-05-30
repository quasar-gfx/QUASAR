#ifndef TEXTURED_MATERIAL_H
#define TEXTURED_MATERIAL_H

#include <Materials/Material.h>

struct UnlitMaterialCreateParams {
    glm::vec3 color = glm::vec3(0.0f);
    std::string diffuseTexturePath = "";
    TextureID diffuseTextureID;
    bool transparent = false;
};

class UnlitMaterial : public Material {
public:
    glm::vec3 color = glm::vec3(0.0f);
    bool transparent = false;

    explicit UnlitMaterial() = default;
    explicit UnlitMaterial(const UnlitMaterialCreateParams &params);

    void bind() override;

    unsigned int getTextureCount() override { return 1; }
};

#endif // TEXTURED_MATERIAL_H
