#include <Renderers/ForwardRenderer.h>

ForwardRenderer::ForwardRenderer(unsigned int width, unsigned int height)
        : gBuffer({ .width = width, .height = height })
        , OpenGLRenderer(width, height) {
}

void ForwardRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
}

RenderStats ForwardRenderer::drawScene(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    gBuffer.bind();
    OpenGLRenderer::drawScene(scene, camera);
    gBuffer.unbind();

    return stats;
}

RenderStats ForwardRenderer::drawLights(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    gBuffer.bind();
    stats = OpenGLRenderer::drawLights(scene, camera);
    gBuffer.unbind();

    return stats;
}

RenderStats ForwardRenderer::drawSkyBox(const Scene &scene, const Camera &camera) {
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
