#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <memory>

#include <glad/glad.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Texture.h>
#include <Primatives/Primatives.h>
#include <Primatives/Model.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights/Lights.h>
#include <FullScreenQuad.h>
#include <RenderTargets/GBuffer.h>

class OpenGLRenderer {
public:
    unsigned int width, height;

    std::shared_ptr<Shader> skyboxShader;

    GeometryBuffer gBuffer;
    FullScreenQuad outputFsQuad;

    explicit OpenGLRenderer() = default;
    ~OpenGLRenderer() = default;

    void init(unsigned int width, unsigned int height);
    void updateDirLightShadow(Scene &scene, Camera &camera);
    void updatePointLightShadows(Scene &scene, Camera &camera);
    void drawSkyBox(Scene &scene, Camera &camera);
    void drawObjects(Scene &scene, Camera &camera);
    void drawToScreen(Shader &screenShader);
    void drawToRenderTarget(Shader &screenShader, RenderTarget &renderTarget);
    void resize(unsigned int width, unsigned int height);

private:
    void drawNode(Scene &scene, Camera &camera, Node* node, glm::mat4 parentTransform, Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
