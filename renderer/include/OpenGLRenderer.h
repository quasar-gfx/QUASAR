#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <memory>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <RenderTargets/GBuffer.h>

class OpenGLRenderer {
public:
    unsigned int width, height;

    GeometryBuffer gBuffer;

    FullScreenQuad outputFsQuad;

    explicit OpenGLRenderer(unsigned int width, unsigned int height);
    ~OpenGLRenderer() = default;

    unsigned int updateDirLightShadow(Scene &scene, Camera &camera);
    unsigned int updatePointLightShadows(Scene &scene, Camera &camera);
    unsigned int drawSkyBox(Scene &scene, Camera &camera);
    unsigned int drawObjects(Scene &scene, Camera &camera);
    void drawToScreen(const Shader &screenShader, const RenderTarget* overrideRenderTarget = nullptr);
    void drawToRenderTarget(const Shader &screenShader, const RenderTarget &renderTarget);
    void resize(unsigned int width, unsigned int height);

private:
    Shader skyboxShader;

    unsigned int drawNode(Scene &scene, Camera &camera, Node* node, const glm::mat4 &parentTransform, bool frustumCull = true, const Material* overrideMaterial = nullptr);
    unsigned int drawNode(Scene &scene, Camera &camera, Node* node, const glm::mat4 &parentTransform, const PointLight* pointLight, const Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
