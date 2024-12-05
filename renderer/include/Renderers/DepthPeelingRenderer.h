#ifndef DEPTH_PEELING_H
#define DEPTH_PEELING_H

#include <Renderers/OpenGLRenderer.h>
#include <RenderTargets/GBuffer.h>

class DepthPeelingRenderer : public OpenGLRenderer {
public:
    unsigned int maxLayers;
    float viewBoxSize = 0.5f;
    float edpDelta = 0.002f;

    GeometryBuffer gBuffer;
    std::vector<GeometryBuffer*> peelingLayers;

    DepthPeelingRenderer(const Config &config, unsigned int maxLayers = 4, bool edp = false);
    ~DepthPeelingRenderer() = default;

    virtual void setScreenShaderUniforms(const Shader &screenShader) override;

    virtual void resize(unsigned int width, unsigned int height) override;

    virtual void beginRendering() override;
    virtual void endRendering() override;

    virtual RenderStats drawScene(const Scene &scene, const Camera &camera, uint32_t clearMask) override;
    virtual RenderStats drawSkyBox(const Scene &scene, const Camera &camera) override;
    virtual RenderStats drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    RenderStats compositeLayers();

    void setViewBoxSize(float viewBoxSize) {
        this->viewBoxSize = viewBoxSize;
    }

private:
    bool edp;
    Shader compositeLayersShader;
};

#endif // DEPTH_PEELING_H
