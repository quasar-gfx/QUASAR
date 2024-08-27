#ifndef FORWARD_RENDERER_H
#define FORWARD_RENDERER_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/GBuffer.h>

class ForwardRenderer : public OpenGLRenderer {
public:
    GeometryBuffer gBuffer;

    ForwardRenderer(const Config &config);
    ~ForwardRenderer() = default;

    void setScreenShaderUniforms(const Shader &screenShader) override;

    void resize(unsigned int width, unsigned int height) override;

    void beginRendering() override;
    void endRendering() override;
};

#endif // FORWARD_RENDERER_H
