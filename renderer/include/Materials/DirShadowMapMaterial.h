#ifndef DIR_SHADOW_MAP_MATERIAL_H
#define DIR_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

class DirShadowMapMaterial : public ShadowMapMaterial {
public:
    explicit DirShadowMapMaterial() {
        ShaderDataCreateParams dirShadowMapParams{
            .vertexCodeData = SHADER_DIRSHADOW_VERT,
            .vertexCodeSize = SHADER_DIRSHADOW_VERT_len,
            .fragmentCodeData = SHADER_DIRSHADOW_FRAG,
            .fragmentCodeSize = SHADER_DIRSHADOW_FRAG_len
        };
        shader = std::make_shared<Shader>(dirShadowMapParams);
    }

    void bind() override {
        shader->bind();
    }
};

#endif // DIR_SHADOW_MAP_MATERIAL_H
