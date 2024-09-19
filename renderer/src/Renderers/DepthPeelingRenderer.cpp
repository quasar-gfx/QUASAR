#include <Renderers/DepthPeelingRenderer.h>

DepthPeelingRenderer::DepthPeelingRenderer(const Config &config, unsigned int maxLayers)
        : gBuffer({ .width = config.width, .height = config.height })
        , maxLayers(maxLayers)
        , compositeLayersShader({
            .vertexCodeData = SHADER_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_COMPOSITELAYERS_FRAG,
            .fragmentCodeSize = SHADER_COMPOSITELAYERS_FRAG_len,
            .defines = {
                "#define MAX_LAYERS " + std::to_string(maxLayers)
            }
        })
        , OpenGLRenderer(config) {

    for (int i = 0; i < maxLayers; i++) {
        peelingLayers.push_back(new GeometryBuffer({ .width = config.width, .height = config.height }));
    }
}

void DepthPeelingRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
    for (auto layer : peelingLayers) {
        layer->resize(width, height);
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

    for (int i = 0; i < maxLayers; i++) {
        auto& currentGBuffer = peelingLayers[i];

        currentGBuffer->bind();
        if (clearMask != 0) {
            glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
            glClear(clearMask);
        }

        Texture* prevDepthMap = nullptr;
        if (i >= 1) {
            prevDepthMap = &peelingLayers[i-1]->depthStencilBuffer;
        }

        for (auto& child : scene.children) {
            stats += drawNode(scene, camera, child, glm::mat4(1.0f), true, nullptr, prevDepthMap);
        }

        currentGBuffer->unbind();
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawSkyBox(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    for (auto layer : peelingLayers) {
        layer->bind();
        stats += OpenGLRenderer::drawSkyBoxImpl(scene, camera);
        layer->unbind();
    }

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

    beginRendering();

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    compositeLayersShader.bind();

    for (int i = 0; i < maxLayers; i++) {
        compositeLayersShader.setTexture("peelingLayers[" + std::to_string(i) + "]", peelingLayers[i]->colorBuffer, i);
    }

    stats += outputFsQuad.draw();

    endRendering();

    return stats;
}
