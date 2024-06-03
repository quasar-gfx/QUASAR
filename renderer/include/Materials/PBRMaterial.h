#ifndef PBR_MATERIAL_H
#define PBR_MATERIAL_H

#include <Materials/Material.h>

struct PBRMaterialCreateParams {
    glm::vec3 color = glm::vec3(-1.0f);
    float opacity = -1.0f;
    float metallic = -1.0f;
    float roughness = -1.0f;
    std::string albedoTexturePath = "";
    std::string normalTexturePath = "";
    std::string metallicTexturePath = "";
    std::string roughnessTexturePath = "";
    std::string aoTexturePath = "";
    TextureID albedoTextureID;
    TextureID normalTextureID;
    TextureID metallicTextureID;
    TextureID roughnessTextureID;
    TextureID aoTextureID;
    bool metalRoughnessCombined = false;
    bool transparent = false;
};

class PBRMaterial : public Material {
public:
    glm::vec3 color = glm::vec3(-1.0f);
    float opacity = -1.0f;
    float metallic = -1.0f;
    float roughness = -1.0f;
    bool transparent = false;

    bool metalRoughnessCombined = false;

    explicit PBRMaterial() = default;
    explicit PBRMaterial(const PBRMaterialCreateParams &params);

    void bind() override;

    unsigned int getTextureCount() override { return 5; }
};

#endif // PBR_MATERIAL_H
