#include <HoleFiller.h>
#include <shaders_common.h>

namespace quasar {

HoleFiller::HoleFiller()
    : shader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_COMMON_HOLE_FILLER_FRAG,
        .fragmentCodeSize = SHADER_COMMON_HOLE_FILLER_FRAG_len,
    })
{}

void HoleFiller::enableTonemapping(bool enable) {
    shader.bind();
    shader.setBool("tonemap", enable);
}

void HoleFiller::setDepthThreshold(float depthThreshold) {
    shader.bind();
    shader.setFloat("depthThreshold", depthThreshold);
}

RenderStats HoleFiller::drawToScreen(OpenGLRenderer& renderer) {
    renderer.setScreenShaderUniforms(shader);
    return renderer.drawToScreen(shader);
}

RenderStats HoleFiller::drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) {
    renderer.setScreenShaderUniforms(shader);
    return renderer.drawToRenderTarget(shader, rt);
}

} // namespace quasar
