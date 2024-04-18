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
#include <Framebuffer.h>
#include <FullScreenQuad.h>

class OpenGLRenderer {
public:
    unsigned int width, height;

    FullScreenQuad outputFsQuad;
    GeometryBuffer gBuffer;

    explicit OpenGLRenderer() = default;
    ~OpenGLRenderer() = default;

    void init(unsigned int width, unsigned int height);
    void updateDirLightShadow(Scene &scene, Camera &camera);
    void updatePointLightShadows(Scene &scene, Camera &camera);
    void drawSkyBox(Shader &shader, Scene &scene, Camera &camera);
    void drawObjects(Scene &scene, Camera &camera);
    void drawToScreen(Shader &screenShader, unsigned int screenWidth, unsigned int screenHeight);

private:
    void drawNode(Scene &scene, Camera &camera, Node* node, glm::mat4 parentTransform, Material* overrideMaterial = nullptr);
};

#endif // OPENGL_RENDERER_H
