#ifndef FORWARD_RENDERER_H
#define FORWARD_RENDERER_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/FrameRenderTarget.h>

namespace quasar {

class ForwardRenderer : public OpenGLRenderer {
public:
    bool multiSampled = false;

    FrameRenderTarget frameRT;
#if !defined(__APPLE__) && !defined(__ANDROID__)
    FrameRenderTarget frameRT_MS;
#endif

    ForwardRenderer(const Config& config);
    ~ForwardRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader& screenShader) override;

    virtual void resize(uint width, uint height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
};

} // namespace quasar

#endif // FORWARD_RENDERER_H
