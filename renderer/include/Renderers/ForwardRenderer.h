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

    RenderStats drawScene(const Scene &scene, const PerspectiveCamera &camera, uint32_t clearMask) override;
    RenderStats drawLights(const Scene &scene, const PerspectiveCamera &camera) override;
    RenderStats drawSkyBox(const Scene &scene, const PerspectiveCamera &camera) override;
};

#endif // FORWARD_RENDERER_H
