#ifndef DEFERRED_LIGHTING_MATERIAL_H
#define DEFERRED_LIGHTING_MATERIAL_H

#include <RenderTargets/DeferredGBuffer.h>
#include <Materials/Material.h>

class DeferredLightingMaterial : public Material {
public:
    DeferredLightingMaterial();
    ~DeferredLightingMaterial();

    void bindGBuffer(const DeferredGBuffer &gBuffer) const;
    void bindCamera(const Camera &camera) const;

    void bind() const override {
        shader->bind();
    }

    Shader* getShader() const override {
        return shader;
    }

    unsigned int getTextureCount() const override { return 7; }

    static Shader* shader;
};

#endif // DEFERRED_LIGHTING_MATERIAL_H
