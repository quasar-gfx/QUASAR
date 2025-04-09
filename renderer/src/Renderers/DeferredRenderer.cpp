#include <Renderers/DeferredRenderer.h>

using namespace quasar;

DeferredRenderer::DeferredRenderer(const Config &config)
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
        , gBuffer({ .width = config.width, .height = config.height, .multiSampled = false })
#if !defined(__APPLE__) && !defined(__ANDROID__)
        , gBufferMS({
            .width = config.width,
            .height = config.height,
            .multiSampled = true,
            .numSamples = config.pipeline.multiSampleState.numSamples
        })
#endif
        , OpenGLRenderer(config) {
}

void DeferredRenderer::setScreenShaderUniforms(const Shader &screenShader) {
    // set gbuffer texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", outputRT.colorBuffer, 0);
    screenShader.setTexture("screenDepth", outputRT.depthStencilBuffer, 1);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 2);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 3);
}

void DeferredRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    outputRT.resize(width, height);
    gBuffer.resize(width, height);
#if !defined(__APPLE__) && !defined(__ANDROID__)
    gBufferMS.resize(width, height);
#endif
}

void DeferredRenderer::beginRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
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

void DeferredRenderer::endRendering() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (!multiSampled) {
        gBuffer.unbind();
    }
    else {
        gBufferMS.blitToDeferredGBuffer(gBuffer);
        gBufferMS.unbind();
    }
#else
    gBuffer.unbind();
#endif
}

RenderStats DeferredRenderer::drawScene(const Scene &scene, const Camera &camera, uint32_t clearMask) {
    RenderStats stats;

    beginRendering();

    // disable blending
    pipeline.blendState.blendEnabled = false; pipeline.apply();

    // draw all objects in the scene
    stats += drawSceneImpl(scene, camera, clearMask);

    // reenable blending
    pipeline.blendState.blendEnabled = true; pipeline.apply();

    endRendering();

    return stats;
}

RenderStats DeferredRenderer::drawSkyBox(const Scene &scene, const Camera &camera) {
    outputRT.bind();
    RenderStats stats = drawSkyBoxImpl(scene, camera);
    outputRT.unbind();
    return stats;
}

RenderStats DeferredRenderer::drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask) {
    pipeline.apply();

    RenderStats stats;

    // update shadows
    updateDirLightShadow(scene, camera);
    updatePointLightShadows(scene, camera);

    // draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // draw lights for debugging
    stats += drawLights(scene, camera);

    // draw lighting pass
    stats += lightingPass(scene, camera);

    // draw skybox
    stats += drawSkyBox(scene, camera);

    return stats;
}

RenderStats DeferredRenderer::drawObjectsNoLighting(const Scene &scene, const Camera &camera, uint32_t clearMask) {
    pipeline.apply();

    RenderStats stats;

    // draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // draw lighting pass
    stats += lightingPass(scene, camera);

    // draw skybox
    stats += drawSkyBox(scene, camera);

    return stats;
}

RenderStats DeferredRenderer::lightingPass(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    lightingMaterial.bind();
    lightingMaterial.bindGBuffer(gBuffer);
    lightingMaterial.bindCamera(camera);

    scene.bindMaterial(&lightingMaterial);

    if (scene.ambientLight != nullptr) {
        scene.ambientLight->bindMaterial(&lightingMaterial);
    }

    int texIdx = lightingMaterial.getTextureCount() + Scene::numTextures;
    if (scene.directionalLight != nullptr) {
        scene.directionalLight->bindMaterial(&lightingMaterial);
        lightingMaterial.getShader()->setMat4("lightSpaceMatrix", scene.directionalLight->lightSpaceMatrix);
        lightingMaterial.getShader()->setTexture("dirLightShadowMap", scene.directionalLight->shadowMapRenderTarget.depthBuffer, texIdx);
    }
    texIdx++;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];
        pointLight->setChannel(i);
        lightingMaterial.getShader()->setTexture("pointLightShadowMaps[" + std::to_string(i) + "]", pointLight->shadowMapRenderTarget.depthCubeMap, texIdx);
        pointLight->bindMaterial(&lightingMaterial);
        texIdx++;
    }

    lightingMaterial.getShader()->setInt("numPointLights", static_cast<int>(scene.pointLights.size()));

    // copy depth from gBuffer to outputRT
    gBuffer.blitDepthToRenderTarget(outputRT);

    glDepthFunc(GL_LEQUAL);

    outputRT.bind();
    stats += outputFsQuad.draw();
    outputRT.unbind();

    glDepthFunc(GL_LESS);

    return stats;
}

void DeferredRenderer::copyToGBuffer(GBuffer &gBufferDst) {
    gBuffer.blitToGBuffer(gBufferDst); // copy normals, id, and depth
    outputRT.blitToRenderTarget(gBufferDst); // copy color
}
