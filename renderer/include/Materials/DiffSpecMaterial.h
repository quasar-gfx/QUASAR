#ifndef DIFF_SPEC_MATERIAL_H
#define DIFF_SPEC_MATERIAL_H

#include <Materials/Material.h>

struct DiffSpecMaterialCreateParams {
    std::string diffuseTexturePath = "";
    std::string specularTexturePath = "";
    TextureID diffuseTextureID;
    TextureID specularTextureID;
    float shininess = 1.0f;
};

class DiffSpecMaterial : public Material {
public:
    float shininess = 1.0f;

    explicit DiffSpecMaterial() = default;
    explicit DiffSpecMaterial(const DiffSpecMaterialCreateParams &params);

    void bind() override;

    unsigned int getTextureCount() override { return 2; }
};

#endif // DIFF_SPEC_MATERIAL_H
