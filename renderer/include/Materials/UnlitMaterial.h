#ifndef TEXTURED_MATERIAL_H
#define TEXTURED_MATERIAL_H

#include <Materials/Material.h>

struct UnlitMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    std::string diffuseTexturePath = "";
    Texture* diffuseTexture;
};

class UnlitMaterial : public Material {
public:
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;

    UnlitMaterial() = default;
    UnlitMaterial(const UnlitMaterialCreateParams &params);
    ~UnlitMaterial();

    void bind() const override;

    Shader* getShader() const override {
        return shader;
    }

    unsigned int getTextureCount() const override { return 1; }

    static Shader* shader;
};

#endif // TEXTURED_MATERIAL_H
