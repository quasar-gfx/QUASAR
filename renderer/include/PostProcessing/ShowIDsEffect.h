#ifndef SHOW_IDS_H
#define SHOW_IDS_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

class ShowIDsEffect : public PostProcessingEffect {
public:
    ShowIDsEffect()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DISPLAYIDS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DISPLAYIDS_FRAG_len
        }) {}

    void showObjectIDs(bool show) {
        shader.bind();
        shader.setBool("showObjectIDs", show);
    }

    void drawToScreen(OpenGLRenderer& renderer) override {
        renderer.drawToScreen(shader);
    }

    void drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase &rt) override {
        renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;
};

#endif // SHOW_IDS_H
