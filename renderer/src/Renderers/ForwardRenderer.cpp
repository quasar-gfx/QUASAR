#include <Renderers/ForwardRenderer.h>

ForwardRenderer::ForwardRenderer(const Config &config)
        : gBuffer({ .width = config.width, .height = config.height })
        , OpenGLRenderer(config) {
}

void ForwardRenderer::setScreenShaderUniforms(const Shader &screenShader) {
    // set gbuffer texture uniforms
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 0);
    screenShader.setTexture("screenDepth", gBuffer.depthBuffer, 1);
    screenShader.setTexture("screenPositions", gBuffer.positionBuffer, 2);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 3);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 4);
}

void ForwardRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
}

void ForwardRenderer::beginRendering() {
    gBuffer.bind();
}

void ForwardRenderer::endRendering() {
    gBuffer.unbind();
}
