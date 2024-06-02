#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <memory>

#include <glad/glad.h>

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

    void updateDirLightShadow(Scene &scene, Camera &camera);
    void updatePointLightShadows(Scene &scene, Camera &camera);
    unsigned int drawSkyBox(Scene &scene, Camera &camera);
    unsigned int drawObjects(Scene &scene, Camera &camera);
    void drawToScreen(Shader &screenShader, RenderTarget* overrideRenderTarget = nullptr);
    void drawToRenderTarget(Shader &screenShader, RenderTarget &renderTarget);
    void resize(unsigned int width, unsigned int height);

private:
    Shader skyboxShader;

    unsigned int drawNode(Scene &scene, Camera &camera, Node* node, glm::mat4 parentTransform, Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
