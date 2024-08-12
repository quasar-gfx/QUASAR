#include <Renderers/DepthPeelingRenderer.h>

DepthPeelingRenderer::DepthPeelingRenderer(unsigned int width, unsigned int height)
        : gBuffer({ .width = width, .height = height })
        , OpenGLRenderer(width, height) {

    for (int i = 0; i < maxLayers; i++) {
        peelingLayers.push_back(new GeometryBuffer({ .width = width, .height = height }));
    }
}

void DepthPeelingRenderer::resize(unsigned int width, unsigned int height) {
    OpenGLRenderer::resize(width, height);
    gBuffer.resize(width, height);
    for (auto layer : peelingLayers) {
        layer->resize(width, height);
    }
}

RenderStats DepthPeelingRenderer::drawScene(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    for (int i = 0; i < maxLayers; i++) {
        auto& currentGBuffer = peelingLayers[i];

        currentGBuffer->bind();
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Texture* prevDepthMap = nullptr;
        if (i >= 1) {
            prevDepthMap = &peelingLayers[i-1]->depthBuffer;
        }

        for (auto& child : scene.children) {
            stats += drawNode(scene, camera, child, glm::mat4(1.0f), true, nullptr, prevDepthMap);
        }

        currentGBuffer->unbind();
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawLights(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    gBuffer.bind();
    stats = OpenGLRenderer::drawLights(scene, camera);
    gBuffer.unbind();

    return stats;
}

RenderStats DepthPeelingRenderer::drawSkyBox(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    for (auto layer : peelingLayers) {
        layer->bind();
        stats += OpenGLRenderer::drawSkyBox(scene, camera);
        layer->unbind();
    }

    return stats;
}

void DepthPeelingRenderer::setScreenShaderUniforms(const Shader &screenShader) {
    // set gbuffer texture uniforms
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 0);
    screenShader.setTexture("screenDepth", gBuffer.depthBuffer, 1);
    screenShader.setTexture("screenPositions", gBuffer.positionBuffer, 2);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 3);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 4);

    screenShader.setInt("maxLayers", maxLayers);
    for (int i = 0; i < maxLayers; i++) {
        screenShader.setTexture("peelingLayers[" + std::to_string(i) + "]", peelingLayers[i]->colorBuffer, 5 + i);
    }
}
