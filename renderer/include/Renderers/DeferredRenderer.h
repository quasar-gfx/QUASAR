#ifndef DEFERRED_RENDERER_H
#define DEFERRED_RENDERER_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/RenderTarget.h>
#include <RenderTargets/GBuffer.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Materials/DeferredLightingMaterial.h>

namespace quasar {

class DeferredRenderer : public OpenGLRenderer {
public:
    bool multiSampled = false;

    RenderTarget outputRT;
    GBuffer gBuffer;
#if !defined(__APPLE__) && !defined(__ANDROID__)
    GBuffer gBuffer_MS;
#endif

    DeferredRenderer(const Config& config);
    ~DeferredRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader& screenShader) override;

    virtual void resize(uint width, uint height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawSkyBox(Scene& scene, const Camera& camera, uint32_t clearMask = 0) override;
    virtual RenderStats drawScene(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    virtual void copyToFrameRT(FrameRenderTarget& frameRT);

protected:
    DeferredLightingMaterial lightingMaterial;

    RenderStats lightingPass(Scene& scene, const Camera& camera);
};

} // namespace quasar

#endif // DEFERRED_RENDERER_H
