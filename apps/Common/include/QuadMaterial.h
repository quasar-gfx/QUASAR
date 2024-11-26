#ifndef QUAD_MATERIAL_H
#define QUAD_MATERIAL_H

#include <Materials/Material.h>

struct QuadMapDataPacked {
    unsigned int normalSpherical; // normal converted into spherical coordinates. theta, phi (16 bits each) packed into uint
    float depth; // 32 bits
    unsigned int xy; // x << 16 | y (12 bits each)
    unsigned int offsetSizeFlattened; // offset.xy << 8 (12 bits each) | size << 1 (5 bits) | flattened (1 bit)
}; // 128 bits total

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
    QuadMaterial(const QuadMaterialCreateParams &params);
    ~QuadMaterial();

    void bind() const override;

    Shader* getShader() const override {
        return shader;
    }

    unsigned int getTextureCount() const override { return 1; }

    static Shader* shader;
};

#endif // QUAD_MATERIAL_H
