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

    unsigned int numLayers = 4;

    GeometryBuffer gBuffer;

    GraphicsPipeline pipeline;

    explicit OpenGLRenderer(unsigned int width, unsigned int height);
    ~OpenGLRenderer() = default;

    void setGraphicsPipeline(const GraphicsPipeline &pipeline) { this->pipeline = pipeline; }

    RenderStats updateDirLightShadow(const Scene &scene, const Camera &camera);
    RenderStats updatePointLightShadows(const Scene &scene, const Camera &camera);

    RenderStats drawScene(const Scene &scene, const Camera &camera);
    RenderStats drawLights(const Scene &scene, const Camera &camera);
    RenderStats drawSkyBox(const Scene &scene, const Camera &camera);
    RenderStats drawObjects(const Scene &scene, const Camera &camera);

    RenderStats drawToScreen(const Shader &screenShader, const RenderTarget* overrideRenderTarget = nullptr);
    RenderStats drawToRenderTarget(const Shader &screenShader, const RenderTarget &renderTarget);

    void resize(unsigned int width, unsigned int height);

private:
    Shader skyboxShader;

    FullScreenQuad outputFsQuad;

    RenderStats drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                         bool frustumCull = true, const Material* overrideMaterial = nullptr);
    RenderStats drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                         const PointLight* pointLight, const Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
