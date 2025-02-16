#ifndef SHOW_NORMALS_H
#define SHOW_NORMALS_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

class ShowNormalsEffect : public PostProcessingEffect {
public:
    ShowNormalsEffect()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DISPLAYNORMALS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DISPLAYNORMALS_FRAG_len
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

#endif // SHOW_NORMALS_H
