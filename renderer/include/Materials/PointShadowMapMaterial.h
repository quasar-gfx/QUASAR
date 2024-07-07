#ifndef POINT_SHADOW_MAP_MATERIAL_H
#define POINT_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

class PointShadowMapMaterial : public ShadowMapMaterial {
public:
    explicit PointShadowMapMaterial() {
        ShaderDataCreateParams pointShadowMapParams{
            .vertexCodeData = SHADER_POINTSHADOW_VERT,
            .vertexCodeSize = SHADER_POINTSHADOW_VERT_len,
            .fragmentCodeData = SHADER_POINTSHADOW_FRAG,
            .fragmentCodeSize = SHADER_POINTSHADOW_FRAG_len,
            .geometryData = SHADER_POINTSHADOW_GEOM,
            .geometryDataSize = SHADER_POINTSHADOW_GEOM_len
        };
        shader = std::make_shared<Shader>(pointShadowMapParams);
    }

    void bind() const override {
        shader->bind();
    }
};

#endif // POINT_SHADOW_MAP_MATERIAL_H
