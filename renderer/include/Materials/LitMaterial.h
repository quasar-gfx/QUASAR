#ifndef PBR_MATERIAL_H
#define PBR_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

struct LitMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    Material::AlphaMode alphaMode = Material::AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    glm::vec3 emissiveFactor = glm::vec3(1.0f);
    float metallic = 0.0f;
    float metallicFactor = 1.0f;
    float roughness = 1.0f;
    float roughnessFactor = 1.0f;
    std::string albedoTexturePath = "";
    std::string normalTexturePath = "";
    std::string metallicTexturePath = "";
    std::string roughnessTexturePath = "";
    std::string aoTexturePath = "";
    std::string emissiveTexturePath = "";
    const Texture* albedoTexture;
    const Texture* normalTexture;
    const Texture* metallicTexture;
    const Texture* roughnessTexture;
    const Texture* aoTexture;
    const Texture* emissiveTexture;
    bool metalRoughnessCombined = false;
};

class LitMaterial : public Material {
public:
    glm::vec4 baseColor;
    glm::vec4 baseColorFactor;
    float maskThreshold;
    glm::vec3 emissiveFactor;

    float metallic;
    float metallicFactor;
    float roughness;
    float roughnessFactor;
    bool metalRoughnessCombined;

    LitMaterial() = default;
    LitMaterial(const LitMaterialCreateParams& params);
    ~LitMaterial() = default;

    void bind() const override;

    std::shared_ptr<Shader> getShader() const override {
        return (pipelineMode == RenderPipelineMode::Forward) ? forwardShader : deferredShader;
    }

    uint getTextureCount() const override { return 6; }

    static void setPipelineMode(RenderPipelineMode mode);

    static std::shared_ptr<Shader> shader;
    static std::shared_ptr<Shader> deferredShader;
    static std::shared_ptr<Shader> forwardShader;

    static std::vector<std::string> extraShaderDefines;
};

} // namespace quasar

#endif // PBR_MATERIAL_H
