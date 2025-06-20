#ifndef SHOW_POSITIONS_H
#define SHOW_POSITIONS_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class ShowPositionsEffect : public PostProcessingEffect {
public:
    ShowPositionsEffect()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_SHOW_POSITIONS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_SHOW_POSITIONS_FRAG_len
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

#endif // SHOW_POSITIONS_H
