#ifndef DEFERRED_LIGHTING_MATERIAL_H
#define DEFERRED_LIGHTING_MATERIAL_H

#include <RenderTargets/GBuffer.h>
#include <Materials/Material.h>

namespace quasar {

class DeferredLightingMaterial : public Material {
public:
    DeferredLightingMaterial();
    ~DeferredLightingMaterial();

    void bindGBuffer(const GBuffer &FrameRenderTarget) const;
    void bindCamera(const Camera &camera) const;

    void bind() const override {
        shader->bind();
    }

    Shader* getShader() const override {
        return shader;
    }

    unsigned int getTextureCount() const override { return 6; }

    static Shader* shader;
};

} // namespace quasar

#endif // DEFERRED_LIGHTING_MATERIAL_H
