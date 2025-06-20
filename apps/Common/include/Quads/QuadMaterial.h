#ifndef QUAD_MATERIAL_H
#define QUAD_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

struct QuadMaterialCreateParams {
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;
    std::string baseColorTexturePath = "";
    Texture* baseColorTexture;
};

class QuadMaterial : public Material {
public:
    glm::vec4 baseColor = glm::vec4(1.0f);
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    AlphaMode alphaMode = AlphaMode::OPAQUE;
    float maskThreshold = 0.5f;

    QuadMaterial() = default;
    QuadMaterial(const QuadMaterialCreateParams& params);
    ~QuadMaterial();

    void bind() const override;

    Shader* getShader() const override {
        return shader;
    }

    uint getTextureCount() const override { return 1; }

    static Shader* shader;
};

} // namespace quasar

#endif // QUAD_MATERIAL_H
