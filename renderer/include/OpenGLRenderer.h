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
    void drawSkyBox(Scene &scene, Camera &camera);
    void drawObjects(Scene &scene, Camera &camera);
    void drawToScreen(Shader &screenShader);
    void drawToRenderTarget(Shader &screenShader, RenderTarget &renderTarget);
    void resize(unsigned int width, unsigned int height);

private:
    Shader skyboxShader;

    void drawNode(Scene &scene, Camera &camera, Node* node, glm::mat4 parentTransform, Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
