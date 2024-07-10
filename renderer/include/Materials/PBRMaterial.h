#ifndef PBR_MATERIAL_H
#define PBR_MATERIAL_H

#include <Materials/Material.h>

struct PBRMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    float metallic = -1.0f;
    float metallicFactor = 1.0f;
    float roughness = -1.0f;
    float roughnessFactor = 1.0f;
    unsigned int numPointLights = 4;
    std::string albedoTexturePath = "";
    std::string normalTexturePath = "";
    std::string metallicTexturePath = "";
    std::string roughnessTexturePath = "";
    std::string aoTexturePath = "";
    std::string emissiveTexturePath = "";
    TextureID albedoTextureID;
    TextureID normalTextureID;
    TextureID metallicTextureID;
    TextureID roughnessTextureID;
    TextureID aoTextureID;
    TextureID emissiveTextureID;
    bool metalRoughnessCombined = false;
};

class PBRMaterial : public Material {
public:
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float opacity = 1.0f;
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;

    float metallic = -1.0f;
    float metallicFactor = 1.0f;
    float roughness = -1.0f;
    float roughnessFactor = 1.0f;
    bool metalRoughnessCombined = false;

    explicit PBRMaterial() = default;
    explicit PBRMaterial(const PBRMaterialCreateParams &params);

    void bind() const override;

    unsigned int getTextureCount() const override { return 6; }
};

#endif // PBR_MATERIAL_H
