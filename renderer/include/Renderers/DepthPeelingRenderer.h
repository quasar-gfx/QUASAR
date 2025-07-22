#ifndef DEPTH_PEELING_H
#define DEPTH_PEELING_H

#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/GBuffer.h>
#include <RenderTargets/FrameRenderTarget.h>

namespace quasar {

class DepthPeelingRenderer : public DeferredRenderer {
public:
    uint maxLayers;
    float viewSphereDiameter = 0.5f;
    float edpDelta = 0.001f;

    std::vector<FrameRenderTarget> peelingLayers;

    DepthPeelingRenderer(const Config& config, uint maxLayers = 4, bool edp = false);
    ~DepthPeelingRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader& screenShader) override;

    virtual void resize(uint width, uint height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawScene(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawSkyBox(Scene& scene, const Camera& camera) override;
    virtual RenderStats drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    RenderStats compositeLayers();

    void setViewSphereDiameter(float viewSphereDiameter) {
        this->viewSphereDiameter = viewSphereDiameter;
    }

private:
    bool edp;
    Shader compositeLayersShader;
};

} // namespace quasar

#endif // DEPTH_PEELING_H
