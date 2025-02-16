#ifndef SHOW_DEPTH_H
#define SHOW_DEPTH_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

class ShowDepthEffect : public PostProcessingEffect {
public:
    ShowDepthEffect(Camera &camera)
        : camera(camera)
        , shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DISPLAYDEPTH_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DISPLAYDEPTH_FRAG_len
        }) {}

    void drawToScreen(OpenGLRenderer& renderer) override {
        shader.bind();
        shader.setFloat("near", camera.getNear());
        shader.setFloat("far", camera.getFar());
        renderer.drawToScreen(shader);
    }

    void drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase &rt) override {
        shader.bind();
        shader.setFloat("near", camera.getNear());
        shader.setFloat("far", camera.getFar());
        renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;

    Camera &camera;
};

#endif // SHOW_DEPTH_H
