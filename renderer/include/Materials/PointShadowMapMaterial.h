#ifndef POINT_SHADOW_MAP_MATERIAL_H
#define POINT_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

namespace quasar {

class PointShadowMapMaterial : public ShadowMapMaterial {
public:
    PointShadowMapMaterial();
    ~PointShadowMapMaterial();

    void bind() const override {
        shader->bind();
    }

    Shader* getShader() const override {
        return shader;
    }

    static Shader* shader;
};

} // namespace quasar

#endif // POINT_SHADOW_MAP_MATERIAL_H
