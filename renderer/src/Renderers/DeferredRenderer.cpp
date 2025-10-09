#include <Cameras/VRCamera.h>
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
        .multiSampled = false,
    })
    , gBuffer({
        .width = config.width,
        .height = config.height,
        .multiSampled = false,
    })
#if !defined(__APPLE__) && !defined(__ANDROID__)
    , gBuffer_MS({
        .width = config.width,
        .height = config.height,
        .multiSampled = true,
        .numSamples = config.pipeline.multiSampleState.numSamples,
    })
#endif
    , OpenGLRenderer(config)
{}

void DeferredRenderer::setScreenShaderUniforms(const Shader& screenShader) {
    // Set texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", outputRT.colorTexture, 0);
    screenShader.setTexture("screenDepth", outputRT.depthStencilTexture, 1);
    screenShader.setTexture("screenNormals", gBuffer.normalsTexture, 2);
    screenShader.setTexture("screenPositions", gBuffer.positionTexture, 3);
    screenShader.setTexture("idTexture", gBuffer.idTexture, 4);
}

void DeferredRenderer::resize(uint width, uint height) {
    OpenGLRenderer::resize(width, height);
    outputRT.resize(width, height);
    gBuffer.resize(width, height);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    gBuffer_MS.resize(width, height);
#endif
}

void DeferredRenderer::beginRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (!multiSampled) {
        gBuffer.bind();
    }
    else {
        gBuffer_MS.bind();
    }
#else
    gBuffer.bind();
#endif
}

void DeferredRenderer::endRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (!multiSampled) {
        gBuffer.unbind();
    }
    else {
        gBuffer_MS.blit(gBuffer);
        gBuffer_MS.unbind();
    }
#else
    gBuffer.unbind();
#endif
}

RenderStats DeferredRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;

    beginRendering();
    if (clearMask != 0) {
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(clearMask);
    }

    // Disable blending
    pipeline.blendState.blendEnabled = false; pipeline.apply();

    // Draw all objects in the scene
    stats += drawSceneImpl(scene, camera, clearMask);

    // Reenable blending
    pipeline.blendState.blendEnabled = true; pipeline.apply();

    endRendering();

    return stats;
}

RenderStats DeferredRenderer::drawSkyBox(Scene& scene, const Camera& camera, uint32_t clearMask) {
    outputRT.bind();
    if (clearMask != 0) {
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(clearMask);
    }

    RenderStats stats = drawSkyBoxImpl(scene, camera, clearMask);
    outputRT.unbind();
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

RenderStats DeferredRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;
    if (camera.isVR()) {
        auto* vrCamera = static_cast<const VRCamera*>(&camera);

        pipeline.rasterState.scissorTestEnabled = true;

        // Left eye
        gBuffer.setScissor(0, 0, width / 2, height);
        gBuffer.setViewport(0, 0, width / 2, height);
        stats += drawObjects(scene, vrCamera->left, clearMask);

        // Right eye
        gBuffer.setScissor(width / 2, 0, width / 2, height);
        gBuffer.setViewport(width / 2, 0, width / 2, height);
        stats += drawObjects(scene, vrCamera->right, clearMask);
    }
    else {
        pipeline.apply();

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
    }

    return stats;
}

RenderStats DeferredRenderer::lightingPass(Scene& scene, const Camera& camera) {
    RenderStats stats;

    lightingMaterial.bind();
    lightingMaterial.bindGBuffer(gBuffer);
    lightingMaterial.bindCamera(camera);

    // Update material uniforms with lighting information
    scene.bindMaterial(&lightingMaterial, pointLightsUBO);

    // Copy depth from FrameRenderTarget to outputRT
    gBuffer.blitDepthToRenderTarget(outputRT);

    pipeline.depthState.depthFunc = GL_LEQUAL;
    pipeline.apply();

    outputRT.bind();
    outputRT.clear(GL_COLOR_BUFFER_BIT);
    stats += outputFsQuad.draw();
    outputRT.unbind();

    // Reenable blending
    pipeline.depthState.depthFunc = GL_LESS;
    pipeline.apply();

    return stats;
}

void DeferredRenderer::copyToFrameRT(FrameRenderTarget& frameRT) {
    gBuffer.blit(frameRT); // Copy alpha, normals, id, and depth
    outputRT.blit(frameRT); // Copy color
}
