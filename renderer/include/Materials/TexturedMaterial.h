#ifndef TEXTURED_MATERIAL_H
#define TEXTURED_MATERIAL_H

#include <Materials/Material.h>

struct TexturedMaterialCreateParams {
    std::string diffuseTexturePath = "";
    TextureID diffuseTextureID;
};

class TexturedMaterial : public Material {
public:
    TexturedMaterial() = default;

    TexturedMaterial(const TexturedMaterialCreateParams &params);

    void bind() override;
    void unbind() override;

    void cleanup() override;
};

#endif // TEXTURED_MATERIAL_H
