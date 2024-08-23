#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <Texture.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Cameras/Camera.h>
#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Materials/PBRMaterial.h>
#include <Materials/UnlitMaterial.h>
#include <Lights/Lights.h>

#include <OpenGLAppConfig.h>

class OpenGLRenderer {
public:
    unsigned int width, height;

    GraphicsPipeline pipeline;

    OpenGLRenderer(const Config &config);
    ~OpenGLRenderer() = default;

    void setGraphicsPipeline(const GraphicsPipeline &pipeline) { this->pipeline = pipeline; }

    virtual void setScreenShaderUniforms(const Shader &screenShader) {};

    virtual void resize(unsigned int width, unsigned int height);

    RenderStats updateDirLightShadow(const Scene &scene, const Camera &camera);
    RenderStats updatePointLightShadows(const Scene &scene, const Camera &camera);

    virtual RenderStats drawScene(const Scene &scene, const Camera &camera, uint32_t clearMask);
    virtual RenderStats drawLights(const Scene &scene, const Camera &camera);
    virtual RenderStats drawSkyBox(const Scene &scene, const Camera &camera);
    virtual RenderStats drawObjects(const Scene &scene, const Camera &camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    virtual RenderStats drawToScreen(const Shader &screenShader, const RenderTarget* overrideRenderTarget = nullptr);
    virtual RenderStats drawToRenderTarget(const Shader &screenShader, const RenderTarget &renderTarget);

protected:
    Shader skyboxShader;

    FullScreenQuad outputFsQuad;

    RenderStats drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                         bool frustumCull = true, const Material* overrideMaterial = nullptr, const Texture* prevDepthMap = nullptr);
    RenderStats drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                         const PointLight* pointLight, const Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
