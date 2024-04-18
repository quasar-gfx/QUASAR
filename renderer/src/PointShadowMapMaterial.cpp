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

void PointShadowMapMaterial::unbind() {
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    shader->unbind();
}

void PointShadowMapMaterial::cleanup() {
    for (auto &textureID : textures) {
        if (textureID == 0) continue;
        glDeleteTextures(1, &textureID);
    }
}
