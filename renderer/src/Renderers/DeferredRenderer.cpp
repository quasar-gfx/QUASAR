#include <Renderers/DeferredRenderer.h>

using namespace quasar;

DeferredRenderer::DeferredRenderer(const Config& config)
    : multiSampled(config.pipeline.multiSampleState.multiSampleEnabled)
    , outputRT({
        .width = config.width,
        .height = config.height,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .multiSampled = false
    })
    , frameRT({ .width = config.width, .height = config.height, .multiSampled = false })
#if !defined(__APPLE__) && !defined(__ANDROID__)
    , frameRT_MS({
        .width = config.width,
        .height = config.height,
        .multiSampled = true,
        .numSamples = config.pipeline.multiSampleState.numSamples
    })
#endif
    , OpenGLRenderer(config)
{}

void DeferredRenderer::setScreenShaderUniforms(const Shader& screenShader) {
    // Set FrameRenderTarget texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", outputRT.colorBuffer, 0);
    screenShader.setTexture("screenDepth", outputRT.depthStencilBuffer, 1);
    screenShader.setTexture("screenNormals", frameRT.normalsBuffer, 2);
    screenShader.setTexture("screenPositions", frameRT.positionBuffer, 3);
    screenShader.setTexture("idBuffer", frameRT.idBuffer, 4);
}

void DeferredRenderer::resize(uint width, uint height) {
    OpenGLRenderer::resize(width, height);
    outputRT.resize(width, height);
    frameRT.resize(width, height);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    frameRT_MS.resize(width, height);
#endif
}

void DeferredRenderer::beginRendering() {
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

void DeferredRenderer::endRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (!multiSampled) {
        frameRT.unbind();
    }
    else {
        frameRT_MS.blitToFrameRT(frameRT);
        frameRT_MS.unbind();
    }
#else
    frameRT.unbind();
#endif
}

RenderStats DeferredRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;

    beginRendering();

    // Disable blending
    pipeline.blendState.blendEnabled = false; pipeline.apply();

    // Draw all objects in the scene
    stats += drawSceneImpl(scene, camera, clearMask);

    // Reenable blending
    pipeline.blendState.blendEnabled = true; pipeline.apply();

    endRendering();

    return stats;
}

RenderStats DeferredRenderer::drawSkyBox(Scene& scene, const Camera& camera) {
    outputRT.bind();
    RenderStats stats = drawSkyBoxImpl(scene, camera);
    outputRT.unbind();
    return stats;
}

RenderStats DeferredRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    RenderStats stats;

    // Update shadows
    updateDirLightShadow(scene, camera);
    updatePointLightShadows(scene, camera);

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Draw lights for debugging
    stats += drawLights(scene, camera);

    // Draw lighting pass
    stats += lightingPass(scene, camera);

    // Draw skybox
    stats += drawSkyBox(scene, camera);

    return stats;
}

RenderStats DeferredRenderer::drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    RenderStats stats;

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Draw lighting pass
    stats += lightingPass(scene, camera);

    // Draw skybox
    stats += drawSkyBox(scene, camera);

    return stats;
}

RenderStats DeferredRenderer::lightingPass(Scene& scene, const Camera& camera) {
    RenderStats stats;

    lightingMaterial.bind();
    lightingMaterial.bindGBuffer(frameRT);
    lightingMaterial.bindCamera(camera);

    // Update material uniforms with lighting information
    scene.bindMaterial(&lightingMaterial);

    // Copy depth from FrameRenderTarget to outputRT
    frameRT.blitDepthToRenderTarget(outputRT);

    glDepthFunc(GL_LEQUAL);

    outputRT.bind();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    stats += outputFsQuad.draw();

    outputRT.unbind();

    glDepthFunc(GL_LESS);

    return stats;
}

void DeferredRenderer::copyToFrameRT(FrameRenderTarget& gBufferDst) {
    frameRT.blitToFrameRT(gBufferDst); // copy normals, id, and depth
    outputRT.blitToRenderTarget(gBufferDst); // copy color
}
