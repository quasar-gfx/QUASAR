#include <Renderers/ForwardRenderer.h>

ForwardRenderer::ForwardRenderer(const Config &config)
        : multiSampled(config.pipeline.multiSampleState.multiSampleEnabled)
        , gBuffer({ .width = config.width, .height = config.height, .multiSampled = false })
#ifndef __APPLE__
        , gBufferMS({
            .width = config.width,
            .height = config.height,
            .multiSampled = true,
            .numSamples = config.pipeline.multiSampleState.numSamples
        })
#endif
        , OpenGLRenderer(config) {
}

void ForwardRenderer::setScreenShaderUniforms(const Shader &screenShader) {
    // set gbuffer texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 0);
    screenShader.setTexture("screenDepth", gBuffer.depthStencilBuffer, 1);
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
        stats = drawObjects(scene, vrCamera->left, clearMask);

        // right eye
        gBuffer.setScissor(width / 2, 0, width / 2, height);
        gBuffer.setViewport(width / 2, 0, width / 2, height);
        stats = drawObjects(scene, vrCamera->right, clearMask);
    }
    else {
        stats = OpenGLRenderer::drawObjects(scene, camera, clearMask);
    }

    return stats;
}

void ForwardRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
#ifndef __APPLE__
    gBufferMS.resize(width, height);
#endif
}

void ForwardRenderer::beginRendering() {
#ifndef __APPLE__
    if (!multiSampled) {
        gBuffer.bind();
    }
    else {
        gBufferMS.bind();
    }
#else
    gBuffer.bind();
#endif
}

void ForwardRenderer::endRendering() {
#ifndef __APPLE__
    if (!multiSampled) {
        gBuffer.unbind();
    }
    else {
        gBufferMS.blitToGBuffer(gBuffer);
        gBufferMS.unbind();
    }
#else
    gBuffer.unbind();
#endif
}
