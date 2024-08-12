#ifndef DEPTH_PEELING_H
#define DEPTH_PEELING_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/GBuffer.h>

class DepthPeelingRenderer : public OpenGLRenderer {
public:
    unsigned int maxLayers = 4;

    GeometryBuffer gBuffer;
    std::vector<GeometryBuffer*> peelingLayers;

    explicit DepthPeelingRenderer(unsigned int width, unsigned int height);
    ~DepthPeelingRenderer() = default;

    void setScreenShaderUniforms(const Shader &screenShader) override;

    void resize(unsigned int width, unsigned int height) override;

    RenderStats drawScene(const Scene &scene, const Camera &camera) override;
    RenderStats drawLights(const Scene &scene, const Camera &camera) override;
    RenderStats drawSkyBox(const Scene &scene, const Camera &camera) override;
};

#endif // DEPTH_PEELING_H
