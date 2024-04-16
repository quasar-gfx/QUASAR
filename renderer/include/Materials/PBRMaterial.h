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

    PBRMaterial() = default;

    PBRMaterial(const PBRMaterialCreateParams &params);

    void bind(Shader &shader) override;
    void unbind() override;

    void cleanup() override;

    static const unsigned int numTextures = 5;
};

#endif // PBR_MATERIAL_H
