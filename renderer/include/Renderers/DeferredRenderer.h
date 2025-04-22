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
    GBuffer frameRT;
#if !defined(__APPLE__) && !defined(__ANDROID__)
    GBuffer frameRT_MS;
#endif

    DeferredRenderer(const Config &config);
    ~DeferredRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader &screenShader) override;

    virtual void resize(unsigned int width, unsigned int height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawScene(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawSkyBox(const Scene &scene, const Camera &camera) override;
    virtual RenderStats drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjectsNoLighting(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    virtual void copyToFrameRT(FrameRenderTarget &gBufferDst);

protected:
    DeferredLightingMaterial lightingMaterial;

    RenderStats lightingPass(const Scene &scene, const Camera &camera);
};

} // namespace quasar

#endif // DEFERRED_RENDERER_H
