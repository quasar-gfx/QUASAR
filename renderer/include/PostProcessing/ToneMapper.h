#ifndef TONE_MAP_EFFECT_H
#define TONE_MAP_EFFECT_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

class ToneMapper : public PostProcessingEffect {
public:
    ToneMapper() = default;

    void setExposure(float exposure) {
        shader.bind();
        shader.setFloat("exposure", exposure);
    }

    void enableToneMapping(bool enable) {
        shader.bind();
        shader.setBool("toneMap", enable);
    }

    void drawToScreen(OpenGLRenderer& renderer) override {
        renderer.drawToScreen(shader);
    }

    void drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase &rt) override {
        renderer.drawToRenderTarget(shader, rt);
    }

// private:
    ToneMapShader shader;
};

#endif // TONE_MAP_EFFECT_H
