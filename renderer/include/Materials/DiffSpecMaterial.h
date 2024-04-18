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

    DiffSpecMaterial() = default;

    DiffSpecMaterial(const DiffSpecMaterialCreateParams &params);

    void bind() override;
    void unbind() override;

    void cleanup() override;

    static const unsigned int numTextures = 2;
};

#endif // DIFF_SPEC_MATERIAL_H
