#include <Materials/DirShadowMapMaterial.h>

DirShadowMapMaterial::DirShadowMapMaterial() {
    ShaderCreateParams dirShadowMapParams{
        .vertexData = SHADER_DIRSHADOW_VERT,
        .vertexDataSize = SHADER_DIRSHADOW_VERT_len,
        .fragmentData = SHADER_DIRSHADOW_FRAG,
        .fragmentDataSize = SHADER_DIRSHADOW_FRAG_len
    };
    shader = std::make_shared<Shader>(dirShadowMapParams);
}

void DirShadowMapMaterial::bind() {
    shader->bind();
}
