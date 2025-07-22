#ifndef PBR_MATERIAL_H
#define PBR_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

struct LitMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    glm::vec3 emissiveFactor = glm::vec3(1.0f);
    float metallic = 0.0f;
    float metallicFactor = 1.0f;
    float roughness = 1.0f;
    float roughnessFactor = 1.0f;
    std::string albedoTexturePath = "";
    std::string normalTexturePath = "";
    std::string metallicTexturePath = "";
    std::string roughnessTexturePath = "";
    std::string aoTexturePath = "";
    std::string emissiveTexturePath = "";
    Texture* albedoTexture;
    Texture* normalTexture;
    Texture* metallicTexture;
    Texture* roughnessTexture;
    Texture* aoTexture;
    Texture* emissiveTexture;
    bool metalRoughnessCombined = false;
};

class LitMaterial : public Material {
public:
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    glm::vec3 emissiveFactor = glm::vec3(1.0f);

    float metallic = 0.0f;
    float metallicFactor = 1.0f;
    float roughness = 1.0f;
    float roughnessFactor = 1.0f;
    bool metalRoughnessCombined = false;

    LitMaterial() = default;
    LitMaterial(const LitMaterialCreateParams& params);
    ~LitMaterial();

    void bind() const override;

    Shader* getShader() const override {
        return shader;
    }

    uint getTextureCount() const override { return 6; }

    static Shader* shader;

    static std::vector<std::string> extraShaderDefines;
};

} // namespace quasar

#endif // PBR_MATERIAL_H
