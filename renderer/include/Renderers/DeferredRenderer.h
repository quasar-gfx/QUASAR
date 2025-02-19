#ifndef DEFERRED_RENDERER_H
#define DEFERRED_RENDERER_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/RenderTarget.h>
#include <RenderTargets/DeferredGBuffer.h>
#include <RenderTargets/GBuffer.h>

#include <Materials/DeferredLightingMaterial.h>

class DeferredRenderer : public OpenGLRenderer {
public:
    bool multiSampled = false;

    RenderTarget outputRT;
    DeferredGBuffer gBuffer;
#ifndef __APPLE__
    DeferredGBuffer gBufferMS;
#endif

    DeferredRenderer(const Config &config);
    ~DeferredRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader &screenShader) override;

    virtual void resize(unsigned int width, unsigned int height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawSkyBox(const Scene &scene, const Camera &camera);
    virtual RenderStats drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    void copyToGBuffer(GBuffer &gBufferDst);

protected:
    DeferredLightingMaterial lightingMaterial;

    RenderStats lightingPass(const Scene &scene, const Camera &camera);
};

#endif // DEFERRED_RENDERER_H
