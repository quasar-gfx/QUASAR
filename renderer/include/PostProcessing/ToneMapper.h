#ifndef TONE_MAP_EFFECT_H
#define TONE_MAP_EFFECT_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class ToneMapper : public PostProcessingEffect {
public:
    ToneMapper() = default;

    void enableToneMapping(bool enable) {
        shader.bind();
        shader.setBool("toneMap", enable);
    }

    void setExposure(float exposure) {
        shader.bind();
        shader.setFloat("exposure", exposure);
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase &rt) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    ToneMapShader shader;
};

} // namespace quasar

#endif // TONE_MAP_EFFECT_H
