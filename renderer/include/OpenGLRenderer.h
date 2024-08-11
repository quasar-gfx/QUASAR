#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <Texture.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Materials/PBRMaterial.h>
#include <Materials/UnlitMaterial.h>
#include <Lights/Lights.h>
#include <RenderTargets/GBuffer.h>

#include <GraphicsPipeline.h>

class OpenGLRenderer {
public:
    unsigned int width, height;

    unsigned int maxLayers = 4;

    GeometryBuffer gBuffer;
    std::vector<GeometryBuffer*> peelingLayers;

    GraphicsPipeline pipeline;

    explicit OpenGLRenderer(unsigned int width, unsigned int height);
    ~OpenGLRenderer() = default;

    void setGraphicsPipeline(const GraphicsPipeline &pipeline) { this->pipeline = pipeline; }

    void resize(unsigned int width, unsigned int height);

    virtual RenderStats updateDirLightShadow(const Scene &scene, const Camera &camera);
    virtual RenderStats updatePointLightShadows(const Scene &scene, const Camera &camera);

    virtual RenderStats drawScene(const Scene &scene, const Camera &camera);
    virtual RenderStats drawLights(const Scene &scene, const Camera &camera);
    virtual RenderStats drawSkyBox(const Scene &scene, const Camera &camera);
    virtual RenderStats drawObjects(const Scene &scene, const Camera &camera);

    virtual RenderStats drawToScreen(const Shader &screenShader, const RenderTarget* overrideRenderTarget = nullptr);
    virtual RenderStats drawToRenderTarget(const Shader &screenShader, const RenderTarget &renderTarget);

private:
    Shader skyboxShader;

    FullScreenQuad outputFsQuad;

    RenderStats drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                         bool frustumCull = true, const Material* overrideMaterial = nullptr, const Texture* prevDepthMap = nullptr);
    RenderStats drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                         const PointLight* pointLight, const Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
