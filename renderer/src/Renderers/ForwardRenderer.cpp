#include <Renderers/ForwardRenderer.h>

ForwardRenderer::ForwardRenderer(const Config &config)
        : gBuffer({ .width = config.width, .height = config.height })
        , OpenGLRenderer(config) {
}

void ForwardRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
}

RenderStats ForwardRenderer::drawScene(const Scene &scene, const PerspectiveCamera &camera, uint32_t clearMask) {
    RenderStats stats;

    gBuffer.bind();
    stats = OpenGLRenderer::drawScene(scene, camera, clearMask);
    gBuffer.unbind();

    return stats;
}

RenderStats ForwardRenderer::drawLights(const Scene &scene, const PerspectiveCamera &camera) {
    RenderStats stats;

    gBuffer.bind();
    stats = OpenGLRenderer::drawLights(scene, camera);
    gBuffer.unbind();

    return stats;
}

RenderStats ForwardRenderer::drawSkyBox(const Scene &scene, const PerspectiveCamera &camera) {
    RenderStats stats;

    gBuffer.bind();
    stats = OpenGLRenderer::drawSkyBox(scene, camera);
    gBuffer.unbind();

    return stats;
}

void ForwardRenderer::setScreenShaderUniforms(const Shader &screenShader) {
    // set gbuffer texture uniforms
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 0);
    screenShader.setTexture("screenDepth", gBuffer.depthBuffer, 1);
    screenShader.setTexture("screenPositions", gBuffer.positionBuffer, 2);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 3);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 4);
}
