#ifndef DIR_SHADOW_MAP_MATERIAL_H
#define DIR_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

namespace quasar {

class DirShadowMapMaterial : public ShadowMapMaterial {
public:
    DirShadowMapMaterial();
    ~DirShadowMapMaterial();

    void bind() const override {
        shader->bind();
    }

    Shader* getShader() const override {
        return shader;
    }

    static Shader* shader;
};

} // namespace quasar

#endif // DIR_SHADOW_MAP_MATERIAL_H
