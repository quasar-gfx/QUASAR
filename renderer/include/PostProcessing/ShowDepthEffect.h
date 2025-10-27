#ifndef SHOW_DEPTH_H
#define SHOW_DEPTH_H

#include <Shaders/TonemapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class ShowDepthEffect : public PostProcessingEffect {
public:
    ShowDepthEffect(Camera& camera, float depthMultiplier = 1.0f)
        : camera(camera)
        , shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_SHOW_DEPTH_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_SHOW_DEPTH_FRAG_len,
        })
    {
        shader.bind();
        shader.setFloat("depthMultiplier", depthMultiplier);
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        shader.bind();
        shader.setFloat("near", camera.getNear());
        shader.setFloat("far", camera.getFar());
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        shader.bind();
        shader.setFloat("near", camera.getNear());
        shader.setFloat("far", camera.getFar());
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;

    Camera& camera;
};

} // namespace quasar

#endif // SHOW_DEPTH_H
