#ifndef QUAD_MATERIAL_H
#define QUAD_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

struct QuadMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::TRANSPARENT;
    std::string baseColorTexturePath = "";
    const Texture* baseColorTexture;
    std::string alphaTexturePath = "";
    const Texture* alphaTexture;
};

class QuadMaterial : public Material {
public:
    glm::vec4 baseColor;
    glm::vec4 baseColorFactor;
    AlphaMode alphaMode;

    QuadMaterial() = default;
    QuadMaterial(const QuadMaterialCreateParams& params);
    ~QuadMaterial() = default;

    void bind() const override;

    std::shared_ptr<Shader> getShader() const override {
        return shader;
    }

    uint getTextureCount() const override { return 1; }

    static std::shared_ptr<Shader> shader;
};

} // namespace quasar

#endif // QUAD_MATERIAL_H
