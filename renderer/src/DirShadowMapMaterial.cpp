#include <Materials/DirShadowMapMaterial.h>

DirShadowMapMaterial::DirShadowMapMaterial() {
    ShaderCreateParams dirShadowMapParams{
        .vertexCodeData = SHADER_DIRSHADOW_VERT,
        .vertexCodeSize = SHADER_DIRSHADOW_VERT_len,
        .fragmentCodeData = SHADER_DIRSHADOW_FRAG,
        .fragmentCodeSize = SHADER_DIRSHADOW_FRAG_len
    };
    shader = std::make_shared<Shader>(dirShadowMapParams);
}

void DirShadowMapMaterial::bind() {
    shader->bind();
}
