#include <Materials/PointShadowMapMaterial.h>

PointShadowMapMaterial::PointShadowMapMaterial() {
    ShaderCreateParams pointShadowMapParams{
        .vertexData = SHADER_POINTSHADOW_VERT,
        .vertexDataSize = SHADER_POINTSHADOW_VERT_len,
        .fragmentData = SHADER_POINTSHADOW_FRAG,
        .fragmentDataSize = SHADER_POINTSHADOW_FRAG_len,
        .geometryData = SHADER_POINTSHADOW_GEOM,
        .geometryDataSize = SHADER_POINTSHADOW_GEOM_len
    };
    shader = std::make_shared<Shader>(pointShadowMapParams);
}

void PointShadowMapMaterial::bind() {
    shader->bind();
}
