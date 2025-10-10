#include <Cameras/VRCamera.h>
#include <Renderers/DepthPeelingRenderer.h>

using namespace quasar;

DepthPeelingRenderer::DepthPeelingRenderer(const Config& config, uint maxLayers, bool edp)
    : maxLayers(maxLayers)
    , edp(edp)
    , DeferredRenderer(config)
    , compositeLayersShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_COMPOSITE_LAYERS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_COMPOSITE_LAYERS_FRAG_len,
        .defines = {
            "#define MAX_LAYERS " + std::to_string(maxLayers)
        }
    })
{
    // Enable depth peeling in shaders
    LitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    UnlitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    if (edp) {
        LitMaterial::extraShaderDefines.push_back("#define EDP");
        UnlitMaterial::extraShaderDefines.push_back("#define EDP");
    }

    RenderTargetCreateParams params {
        .width = config.width,
        .height = config.height,
        .internalFormat = outputRT.colorTexture.internalFormat,
        .format = outputRT.colorTexture.format,
        .type = outputRT.colorTexture.type,
        .wrapS = outputRT.colorTexture.wrapS,
        .wrapT = outputRT.colorTexture.wrapT,
        .minFilter = outputRT.colorTexture.minFilter,
        .magFilter = outputRT.colorTexture.magFilter,
        .multiSampled = outputRT.colorTexture.multiSampled,
    };
    peelingLayers.reserve(maxLayers);
    for (int i = 0; i < maxLayers; i++) {
        peelingLayers.emplace_back(params);
    }
}

void DepthPeelingRenderer::resize(uint width, uint height) {
    DeferredRenderer::resize(width, height);
    for (auto layer : peelingLayers) {
        layer.resize(width, height);
    }
}

void DepthPeelingRenderer::setScreenShaderUniforms(const Shader& screenShader) {
    // Set texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", outputRT.colorTexture, 0);
    screenShader.setTexture("screenDepth", peelingLayers[0].depthStencilTexture, 1);
    screenShader.setTexture("screenNormals", gBuffer.normalsTexture, 2);
    screenShader.setTexture("screenPositions", gBuffer.positionTexture, 3);
    screenShader.setTexture("idTexture", gBuffer.idTexture, 4);
}

RenderStats DepthPeelingRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;

    for (int i = 0; i < maxLayers; i++) {
        beginRendering();
        if (clearMask != 0) {
            gBuffer.clear(clearMask);
        }

        // Disable blending
        pipeline.blendState.blendEnabled = false; pipeline.apply();

        const Texture* prevIDMap = (i >= 1) ? &peelingLayers[i-1].idTexture : nullptr;

        // Set layer index in shaders
        if (LitMaterial::deferredShader != nullptr) {
            LitMaterial::deferredShader->bind();
            LitMaterial::deferredShader->setInt("layerIndex", i);
        }
        if (LitMaterial::forwardShader != nullptr) {
            LitMaterial::forwardShader->bind();
            LitMaterial::forwardShader->setInt("layerIndex", i);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("layerIndex", i);
        }

        // Render scene
        for (auto* child : scene.children) {
            stats += drawNodeImmediate(scene, camera, child, glm::mat4(1.0f), true, nullptr, prevIDMap);
        }

        // Re-enable blending
        pipeline.blendState.blendEnabled = true; pipeline.apply();

        endRendering();

        // Draw lighting pass
        stats += lightingPass(scene, camera);

        // Draw skybox
        stats += drawSkyBox(scene, camera);

        copyToFrameRT(peelingLayers[i]);
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;
    if (camera.isVR()) {
        auto* vrCamera = static_cast<const VRCamera*>(&camera);

        pipeline.rasterState.scissorTestEnabled = true;

        // Left eye
        gBuffer.setScissor(0, 0, width / 2, height);
        gBuffer.setViewport(0, 0, width / 2, height);
        outputRT.setScissor(0, 0, width / 2, height);
        outputRT.setViewport(0, 0, width / 2, height);
        stats += drawObjects(scene, vrCamera->left, clearMask);

        // Right eye
        gBuffer.setScissor(width / 2, 0, width / 2, height);
        gBuffer.setViewport(width / 2, 0, width / 2, height);
        outputRT.setScissor(width / 2, 0, width / 2, height);
        outputRT.setViewport(width / 2, 0, width / 2, height);
        stats += drawObjects(scene, vrCamera->right, clearMask);
    }
    else {
        pipeline.apply();

        if (edp) {
            if (LitMaterial::deferredShader != nullptr) {
                LitMaterial::deferredShader->bind();
                LitMaterial::deferredShader->setInt("height", gBuffer.height);
                LitMaterial::deferredShader->setFloat("E", viewSphereDiameter / 2.0f);
                LitMaterial::deferredShader->setFloat("edpDelta", edpDelta);
            }
            if (LitMaterial::forwardShader != nullptr) {
                LitMaterial::forwardShader->bind();
                LitMaterial::forwardShader->setInt("height", gBuffer.height);
                LitMaterial::forwardShader->setFloat("E", viewSphereDiameter / 2.0f);
                LitMaterial::forwardShader->setFloat("edpDelta", edpDelta);
            }
            if (UnlitMaterial::shader != nullptr) {
                UnlitMaterial::shader->bind();
                UnlitMaterial::shader->setInt("height", gBuffer.height);
                UnlitMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
                UnlitMaterial::shader->setFloat("edpDelta", edpDelta);
            }
        }

        RenderStats stats;

        // Update shadows
        updateDirLightShadow(scene, camera);
        updatePointLightShadows(scene, camera);

        // Draw all objects in the scene
        stats += drawScene(scene, camera, clearMask);

        // Draw lights for debugging
        stats += drawLights(scene, camera);

        // Dont draw skybox here, it's drawn in drawScene

        // Composite layers
        stats += compositeLayers();
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    if (edp) {
        if (LitMaterial::deferredShader != nullptr) {
            LitMaterial::deferredShader->bind();
            LitMaterial::deferredShader->setInt("height", gBuffer.height);
            LitMaterial::deferredShader->setFloat("E", viewSphereDiameter / 2.0f);
            LitMaterial::deferredShader->setFloat("edpDelta", edpDelta);
        }
        if (LitMaterial::forwardShader != nullptr) {
            LitMaterial::forwardShader->bind();
            LitMaterial::forwardShader->setInt("height", gBuffer.height);
            LitMaterial::forwardShader->setFloat("E", viewSphereDiameter / 2.0f);
            LitMaterial::forwardShader->setFloat("edpDelta", edpDelta);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("height", gBuffer.height);
            UnlitMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
            UnlitMaterial::shader->setFloat("edpDelta", edpDelta);
        }
    }

    RenderStats stats;

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Dont draw skybox here, it's drawn in drawScene

    // Composite layers
    stats += compositeLayers();

    return stats;
}

RenderStats DepthPeelingRenderer::compositeLayers() {
    RenderStats stats;

    compositeLayersShader.bind();
    for (int i = 0; i < maxLayers; i++) {
        compositeLayersShader.setTexture(
            "peelingLayersColor[" + std::to_string(i) + "]", peelingLayers[i].colorTexture, i);
        compositeLayersShader.setTexture(
            "peelingLayersAlpha[" + std::to_string(i) + "]", peelingLayers[i].alphaTexture, i + maxLayers);
    }

    outputRT.bind();
    outputRT.clear(GL_COLOR_BUFFER_BIT);
    stats += outputFsQuad->draw();
    outputRT.unbind();

    return stats;
}
