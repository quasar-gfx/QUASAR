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
            .geometryDataSize = SHADER_POINTSHADOW_GEOM_len,
#ifdef GL_ES
            .extensions = {
                "#extension GL_EXT_geometry_shader : enable"
            }
#endif
        };
        shader = std::make_shared<Shader>(pointShadowMapParams);
    }
};

#endif // POINT_SHADOW_MAP_MATERIAL_H
