#ifndef SHOW_IDS_H
#define SHOW_IDS_H

#include <Shaders/ToneMapShader.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class ShowIDsEffect : public PostProcessingEffect {
public:
    ShowIDsEffect()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_SHOW_IDS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_SHOW_IDS_FRAG_len
        }) {}

    void showObjectIDs(bool showObjectID) {
        shader.bind();
        shader.setBool("showObjectIDs", showObjectID);
    }

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

#endif // SHOW_IDS_H
