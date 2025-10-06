#ifndef FXAA_H
#define FXAA_H

#include <Shaders/TonemapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class FXAA : public PostProcessingEffect {
public:
    FXAA()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_FXAA_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_FXAA_FRAG_len,
        })
    {}

    void enableTonemapping(bool enable) {
        shader.bind();
        shader.setBool("tonemap", enable);
    }

    void setExposure(float exposure) {
        shader.bind();
        shader.setFloat("exposure", exposure);
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        shader.bind();
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        shader.bind();
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;
};

} // namespace quasar

#endif // FXAA_H
