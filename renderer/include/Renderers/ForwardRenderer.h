#ifndef FORWARD_RENDERER_H
#define FORWARD_RENDERER_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/GBuffer.h>

class ForwardRenderer : public OpenGLRenderer {
public:
    bool multiSampled = false;

    GBuffer gBuffer;
#ifndef __APPLE__
    GBuffer gBufferMS;
#endif

    ForwardRenderer(const Config &config);
    ~ForwardRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader &screenShader) override;

    virtual void resize(unsigned int width, unsigned int height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
};

#endif // FORWARD_RENDERER_H
