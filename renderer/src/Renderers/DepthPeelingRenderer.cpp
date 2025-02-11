#include <Renderers/DepthPeelingRenderer.h>

DepthPeelingRenderer::DepthPeelingRenderer(const Config &config, unsigned int maxLayers, bool edp)
        : maxLayers(maxLayers)
        , OpenGLRenderer(config)
        , gBuffer({ .width = config.width, .height = config.height })
        , compositeLayersShader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_COMPOSITELAYERS_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_COMPOSITELAYERS_FRAG_len,
            .defines = {
                "#define MAX_LAYERS " + std::to_string(maxLayers)
            }
        })
        , edp(edp) {
    // enable depth peeling in shaders
    PBRMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    UnlitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    if (edp) {
        PBRMaterial::extraShaderDefines.push_back("#define EDP");
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

void DepthPeelingRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
    for (auto layer : peelingLayers) {
        layer.resize(width, height);
    }
}

void DepthPeelingRenderer::setScreenShaderUniforms(const Shader &screenShader) {
    // set gbuffer texture uniforms
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 0);
    screenShader.setTexture("screenDepth", gBuffer.depthStencilBuffer, 1);
    screenShader.setTexture("screenPositions", gBuffer.positionBuffer, 2);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 3);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 4);
}

void DepthPeelingRenderer::beginRendering() {
    gBuffer.bind();
}

void DepthPeelingRenderer::endRendering() {
    gBuffer.unbind();
}

RenderStats DepthPeelingRenderer::drawScene(const Scene &scene, const Camera &camera, uint32_t clearMask) {
    RenderStats stats;

    if (edp) {
        if (PBRMaterial::shader != nullptr) {
            PBRMaterial::shader->bind();
            PBRMaterial::shader->setInt("height", gBuffer.height);
            PBRMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
            PBRMaterial::shader->setFloat("edpDelta", edpDelta);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("height", gBuffer.height);
            UnlitMaterial::shader->setFloat("E", viewSphereDiameter / 2.0f);
            UnlitMaterial::shader->setFloat("edpDelta", edpDelta);
        }
    }

    for (int i = 0; i < maxLayers; i++) {
        auto& gBuffer = peelingLayers[i];

        gBuffer.bind();
        if (clearMask != 0) {
            glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
            glClear(clearMask);
        }

        Texture* prevDepthMap = nullptr;
        if (i >= 1) {
            prevDepthMap = &peelingLayers[i-1].idBuffer;
        }

        // set layer index in shaders
        if (PBRMaterial::shader != nullptr) {
            PBRMaterial::shader->bind();
            PBRMaterial::shader->setInt("layerIndex", i);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("layerIndex", i);
        }

        // render scene
        for (auto& child : scene.rootNode.children) {
            stats += drawNode(scene, camera, child, glm::mat4(1.0f), true, nullptr, prevDepthMap);
        }

        gBuffer.unbind();
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawSkyBox(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    peelingLayers[0].bind();
    stats += OpenGLRenderer::drawSkyBoxImpl(scene, camera);
    peelingLayers[0].unbind();

    return stats;
}

RenderStats DepthPeelingRenderer::drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask) {
    RenderStats stats;
    stats = OpenGLRenderer::drawObjects(scene, camera, clearMask);
    stats += compositeLayers();
    return stats;
}

RenderStats DepthPeelingRenderer::compositeLayers() {
    RenderStats stats;

    compositeLayersShader.bind();
    for (int i = 0; i < maxLayers; i++) {
        compositeLayersShader.setTexture("peelingLayers[" + std::to_string(i) + "]", peelingLayers[i].colorBuffer, i);
    }

    beginRendering();
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    stats += outputFsQuad.draw();
    endRendering();

    return stats;
}
