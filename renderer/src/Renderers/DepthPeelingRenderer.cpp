#include <Renderers/DepthPeelingRenderer.h>

using namespace quasar;

DepthPeelingRenderer::DepthPeelingRenderer(const Config& config, uint maxLayers, bool edp)
    : maxLayers(maxLayers)
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
    , edp(edp)
{
    // Enable depth peeling in shaders
    LitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    UnlitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    if (edp) {
        LitMaterial::extraShaderDefines.push_back("#define EDP");
        UnlitMaterial::extraShaderDefines.push_back("#define EDP");
    }

    peelingLayers.reserve(maxLayers);

    RenderTargetCreateParams params {
        .width = config.width,
        .height = config.height
    };
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

void DepthPeelingRenderer::beginRendering() {
    frameRT.bind();
}

void DepthPeelingRenderer::endRendering() {
    frameRT.unbind();
}

void DepthPeelingRenderer::setScreenShaderUniforms(const Shader& screenShader) {
    // Set FrameRenderTarget texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", outputRT.colorBuffer, 0);
    screenShader.setTexture("screenDepth", outputRT.depthStencilBuffer, 1);
    screenShader.setTexture("screenNormals", frameRT.normalsBuffer, 2);
    screenShader.setTexture("screenPositions", frameRT.positionBuffer, 3);
    screenShader.setTexture("idBuffer", frameRT.idBuffer, 4);
}

RenderStats DepthPeelingRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;

    for (int i = 0; i < maxLayers; i++) {
        beginRendering();
        if (clearMask != 0) {
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(clearMask);
        }

        // Disable blending
        pipeline.blendState.blendEnabled = false; pipeline.apply();

        Texture* prevIDMap = (i >= 1) ? &peelingLayers[i-1].idBuffer : nullptr;

        // Set layer index in shaders
        if (LitMaterial::shader != nullptr) {
            LitMaterial::shader->bind();
            LitMaterial::shader->setInt("layerIndex", i);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("layerIndex", i);
        }

        // Render scene
        for (auto& child : scene.rootNode.children) {
            stats += drawNode(scene, camera, child, glm::mat4(1.0f), true, nullptr, prevIDMap);
        }

        // Reenable blending
        pipeline.blendState.blendEnabled = true; pipeline.apply();

        endRendering();

        // Clear output render target
        outputRT.bind();
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(GL_COLOR_BUFFER_BIT);
        outputRT.unbind();

        // Draw lighting pass
        stats += lightingPass(scene, camera);

        if (i == 0) {
            // Draw skybox only on first layer
            stats += drawSkyBox(scene, camera);
        }

        copyToFrameRT(peelingLayers[i]);
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawSkyBox(Scene& scene, const Camera& camera) {
    outputRT.bind();
    RenderStats stats = drawSkyBoxImpl(scene, camera);
    outputRT.unbind();
    return stats;
}

RenderStats DepthPeelingRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    if (edp) {
        if (LitMaterial::shader != nullptr) {
            LitMaterial::shader->bind();
            LitMaterial::shader->setInt("height", frameRT.height);
            LitMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
            LitMaterial::shader->setFloat("edpDelta", edpDelta);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("height", frameRT.height);
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

    return stats;
}

RenderStats DepthPeelingRenderer::drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    if (edp) {
        if (LitMaterial::shader != nullptr) {
            LitMaterial::shader->bind();
            LitMaterial::shader->setInt("height", frameRT.height);
            LitMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
            LitMaterial::shader->setFloat("edpDelta", edpDelta);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("height", frameRT.height);
            UnlitMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
            UnlitMaterial::shader->setFloat("edpDelta", edpDelta);
        }
    }

    RenderStats stats;

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Draw lighting pass
    stats += lightingPass(scene, camera);

    // Dont draw skybox here, it's drawn in drawScene

    // Composite layers
    stats += compositeLayers();

    return stats;
}

RenderStats DepthPeelingRenderer::compositeLayers() {
    RenderStats stats;

    compositeLayersShader.bind();
    for (int i = 0; i < maxLayers; i++) {
        compositeLayersShader.setTexture("peelingLayers[" + std::to_string(i) + "]", peelingLayers[i].colorBuffer, i);
    }

    outputRT.bind();
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    stats += outputFsQuad.draw();
    outputRT.unbind();

    return stats;
}
