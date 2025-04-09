#include <Materials/DirShadowMapMaterial.h>

using namespace quasar;

Shader* DirShadowMapMaterial::shader = nullptr;

DirShadowMapMaterial::DirShadowMapMaterial() {
    if (shader == nullptr) {
        ShaderDataCreateParams dirShadowMapParams{
            .vertexCodeData = SHADER_BUILTIN_DIRSHADOW_VERT,
            .vertexCodeSize = SHADER_BUILTIN_DIRSHADOW_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DIRSHADOW_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DIRSHADOW_FRAG_len
        };
        shader = new Shader(dirShadowMapParams);
    }
}

DirShadowMapMaterial::~DirShadowMapMaterial() {
    if (shader != nullptr) {
        delete shader;
        shader = nullptr;
    }
}
