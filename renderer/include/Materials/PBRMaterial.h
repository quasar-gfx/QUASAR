#ifndef PBR_MATERIAL_H
#define PBR_MATERIAL_H

#include <Materials/Material.h>

struct PBRMaterialCreateParams {
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
};

class PBRMaterial : public Material {
public:
    float shininess = 1.0f;

    explicit PBRMaterial() = default;
    explicit PBRMaterial(const PBRMaterialCreateParams &params);

    void bind() override;

    unsigned int getTextureCount() override { return 5; }
};

#endif // PBR_MATERIAL_H
