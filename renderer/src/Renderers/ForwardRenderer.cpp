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

RenderStats ForwardRenderer::drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask) {
    RenderStats stats;
    if (camera.isVR()) {
        auto* vrCamera = static_cast<const VRCamera*>(&camera);

        pipeline.rasterState.scissorTestEnabled = true;

        // left eye
        gBuffer.setScissor(0, 0, width / 2, height);
        gBuffer.setViewport(0, 0, width / 2, height);
        stats = drawObjects(scene, vrCamera->left);

        // right eye
        gBuffer.setScissor(width / 2, 0, width / 2, height);
        gBuffer.setViewport(width / 2, 0, width / 2, height);
        stats = drawObjects(scene, vrCamera->right);
    }
    else {
        stats = OpenGLRenderer::drawObjects(scene, camera, clearMask);
    }

    return stats;
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
