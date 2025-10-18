#include <Cameras/VRCamera.h>
#include <Renderers/ForwardRenderer.h>

using namespace quasar;

ForwardRenderer::ForwardRenderer(const Config& config)
    : multiSampled(config.pipeline.multiSampleState.multiSampleEnabled)
    , frameRT({
        .width = config.width,
        .height = config.height,
        .multiSampled = false,
    })
#if !defined(__APPLE__) && !defined(__ANDROID__)
    , frameRT_MS({
        .width = config.width,
        .height = config.height,
        .multiSampled = true,
        .numSamples = config.pipeline.multiSampleState.numSamples,
    })
#endif
    , OpenGLRenderer(config)
{}

void ForwardRenderer::setScreenShaderUniforms(const Shader& screenShader) {
    // Set FrameRenderTarget texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", frameRT.colorTexture, 0);
    screenShader.setTexture("screenDepth", frameRT.depthStencilTexture, 1);
    screenShader.setTexture("screenNormals", frameRT.normalsTexture, 2);
    screenShader.setTexture("screenPositions", frameRT.normalsTexture, 3); // RenderTarget has no position buffer
    screenShader.setTexture("idTexture", frameRT.idTexture, 4);
}

RenderStats ForwardRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;
    if (camera.isVR()) {
        auto* vrCamera = static_cast<const VRCamera*>(&camera);

        pipeline.rasterState.scissorTestEnabled = true;

        // Left eye
        frameRT.setScissor(0, 0, width / 2, height);
        frameRT.setViewport(0, 0, width / 2, height);
        stats = drawObjects(scene, vrCamera->left, clearMask);

        // Right eye
        frameRT.setScissor(width / 2, 0, width / 2, height);
        frameRT.setViewport(width / 2, 0, width / 2, height);
        stats = drawObjects(scene, vrCamera->right, clearMask);
    }
    else {
        stats = OpenGLRenderer::drawObjects(scene, camera, clearMask);
    }

    return stats;
}

void ForwardRenderer::resize(uint width, uint height) {
    OpenGLRenderer::resize(width, height);
    frameRT.resize(width, height);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    frameRT_MS.resize(width, height);
#endif
}

void ForwardRenderer::beginRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (!multiSampled) {
        frameRT.bind();
    }
    else {
        frameRT_MS.bind();
    }
#else
    frameRT.bind();
#endif
}

void ForwardRenderer::endRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (!multiSampled) {
        frameRT.unbind();
    }
    else {
        frameRT_MS.blit(frameRT);
        frameRT_MS.unbind();
    }
#else
    frameRT.unbind();
#endif
}
