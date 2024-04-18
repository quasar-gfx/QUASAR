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

void DirShadowMapMaterial::unbind() {
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    shader->unbind();
}

void DirShadowMapMaterial::cleanup() {
    for (auto &textureID : textures) {
        if (textureID == 0) continue;
        glDeleteTextures(1, &textureID);
    }
}
