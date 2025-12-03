#ifndef UNLIT_MATERIAL_H
#define UNLIT_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

struct UnlitMaterialCreateParams {
    std::string name = "UnlitMaterial" + std::to_string(Material::getNextID());
    glm::vec4 baseColor{1.0f};
    glm::vec4 baseColorFactor{1.0f};
    Material::AlphaMode alphaMode = Material::AlphaMode::OPAQUE;
    float maskThreshold = 0.1f;
    std::string baseColorTexturePath = "";
    const Texture* baseColorTexture;
};

class UnlitMaterial : public Material {
public:
    glm::vec4 baseColor;
    glm::vec4 baseColorFactor;
    float maskThreshold;

    UnlitMaterial() = default;
    UnlitMaterial(const UnlitMaterialCreateParams& params);
    ~UnlitMaterial() = default;

    bool isTransparent() const override;

    void bind() const override;

    std::shared_ptr<Shader> getShader() const override {
        return shader;
    }

    uint getTextureCount() const override { return 1; }

    static std::shared_ptr<Shader> shader;

    static std::vector<std::string> extraShaderDefines;
};

} // namespace quasar

#endif // UNLIT_MATERIAL_H
