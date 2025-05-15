#ifndef SHOW_NORMALS_H
#define SHOW_NORMALS_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class ShowNormalsEffect : public PostProcessingEffect {
public:
    ShowNormalsEffect()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DISPLAYNORMALS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DISPLAYNORMALS_FRAG_len
        }) {}

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;
};

} // namespace quasar

#endif // SHOW_NORMALS_H
