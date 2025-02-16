#ifndef SHOW_POSITIONS_H
#define SHOW_POSITIONS_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

class ShowPositionsEffect : public PostProcessingEffect {
public:
    ShowPositionsEffect()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DISPLAYPOSITIONS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DISPLAYPOSITIONS_FRAG_len
        }) {}

    void drawToScreen(OpenGLRenderer& renderer) override {
        renderer.drawToScreen(shader);
    }

    void drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase &rt) override {
        renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;
};

#endif // SHOW_POSITIONS_H
