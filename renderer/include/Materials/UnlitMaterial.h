#ifndef UNLIT_MATERIAL_H
#define UNLIT_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

struct UnlitMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    Material::AlphaMode alphaMode = Material::AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    std::string baseColorTexturePath = "";
    const Texture* baseColorTexture;
};

class UnlitMaterial : public Material {
public:
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float maskThreshold = 0.5f;

    UnlitMaterial() = default;
    UnlitMaterial(const UnlitMaterialCreateParams& params);
    ~UnlitMaterial() = default;

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
