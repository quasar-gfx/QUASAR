#ifndef BLUR_EDGES_H
#define BLUR_EDGES_H

#include <PostProcessing/PostProcessingEffect.h>

#include <shaders_common.h>

class BlurEdges : public PostProcessingEffect {
public:
    BlurEdges()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_COMMON_BLUREDGES_FRAG,
            .fragmentCodeSize = SHADER_COMMON_BLUREDGES_FRAG_len
        }) {}

    void setDepthThreshold(float depthThreshold) {
        shader.bind();
        shader.setFloat("depthThreshold", depthThreshold);
    }

    void enableToneMapping(bool enable) {
        shader.bind();
        shader.setBool("toneMap", enable);
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
    Shader shader;
};

#endif // BLUR_EDGES_H
