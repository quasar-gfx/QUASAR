#include <Materials/PointShadowMapMaterial.h>

PointShadowMapMaterial::PointShadowMapMaterial() {
    ShaderCreateParams pointShadowMapParams{
        .vertexCodeData = SHADER_POINTSHADOW_VERT,
        .vertexCodeSize = SHADER_POINTSHADOW_VERT_len,
        .fragmentCodeData = SHADER_POINTSHADOW_FRAG,
        .fragmentCodeSize = SHADER_POINTSHADOW_FRAG_len,
        .geometryData = SHADER_POINTSHADOW_GEOM,
        .geometryDataSize = SHADER_POINTSHADOW_GEOM_len
    };
    shader = std::make_shared<Shader>(pointShadowMapParams);
}

void PointShadowMapMaterial::bind() {
    shader->bind();
}
