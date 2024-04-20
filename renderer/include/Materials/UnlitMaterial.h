#ifndef TEXTURED_MATERIAL_H
#define TEXTURED_MATERIAL_H

#include <Materials/Material.h>

struct UnlitMaterialCreateParams {
    std::string diffuseTexturePath = "";
    TextureID diffuseTextureID;
};

class UnlitMaterial : public Material {
public:
    explicit UnlitMaterial() = default;
    explicit UnlitMaterial(const UnlitMaterialCreateParams &params);

    void bind() override;

    unsigned int getTextureCount() override { return 2; }
};

#endif // TEXTURED_MATERIAL_H
