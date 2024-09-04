#ifndef DEPTH_PEELING_H
#define DEPTH_PEELING_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/GBuffer.h>

class DepthPeelingRenderer : public OpenGLRenderer {
public:
    unsigned int maxLayers;

    GeometryBuffer gBuffer;
    std::vector<GeometryBuffer*> peelingLayers;

    DepthPeelingRenderer(const Config &config, unsigned int maxLayers = 8);
    ~DepthPeelingRenderer() = default;

    void setScreenShaderUniforms(const Shader &screenShader) override;

    void resize(unsigned int width, unsigned int height) override;

    void beginRendering() override;
    void endRendering() override;

    RenderStats drawScene(const Scene &scene, const Camera &camera, uint32_t clearMask) override;
    RenderStats drawSkyBox(const Scene &scene, const Camera &camera) override;
    RenderStats drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    RenderStats compositeLayers();

private:
    Shader compositeLayersShader;
};

#endif // DEPTH_PEELING_H
