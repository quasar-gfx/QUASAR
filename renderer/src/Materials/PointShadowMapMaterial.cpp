#include <Materials/PointShadowMapMaterial.h>

using namespace quasar;

Shader* PointShadowMapMaterial::shader = nullptr;

PointShadowMapMaterial::PointShadowMapMaterial() {
    if (shader == nullptr) {
        ShaderDataCreateParams pointShadowMapParams{
            .vertexCodeData = SHADER_BUILTIN_POINTSHADOW_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POINTSHADOW_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_POINTSHADOW_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_POINTSHADOW_FRAG_len,
            .geometryData = SHADER_BUILTIN_POINTSHADOW_GEOM,
            .geometryMetadataSize = SHADER_BUILTIN_POINTSHADOW_GEOM_len,
#ifdef GL_ES
            .extensions = {
                "#extension GL_EXT_geometry_shader : enable"
            }
#endif
        };
        shader = new Shader(pointShadowMapParams);
    }
}

PointShadowMapMaterial::~PointShadowMapMaterial() {
    if (shader != nullptr) {
        delete shader;
        shader = nullptr;
    }
}
