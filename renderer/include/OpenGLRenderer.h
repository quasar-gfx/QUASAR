#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <glad/glad.h>

#include <Shader.h>
#include <ComputeShader.h>
#include <Texture.h>
#include <Primatives.h>
#include <Model.h>
#include <CubeMap.h>
#include <Entity.h>
#include <Scene.h>
#include <Camera.h>
#include <Lights.h>
#include <Framebuffer.h>
#include <FullScreenQuad.h>

class OpenGLRenderer {
public:
    unsigned int width, height;

    FullScreenQuad outputFsQuad;
    GeometryBuffer gBuffer;

    OpenGLRenderer() = default;
    ~OpenGLRenderer() = default;

    void init(unsigned int width, unsigned int height);
    void updateDirLightShadowMap(Shader &shader, Scene &scene, Camera &camera);
    void updatePointLightShadowMaps(Shader &shader, Scene &scene, Camera &camera);
    void drawSkyBox(Shader &shader, Scene &scene, Camera &camera);
    void drawObjects(Shader &shader, Scene &scene, Camera &camera);
    void drawToScreen(Shader &screenShader, unsigned int screenWidth, unsigned int screenHeight);

private:
    void drawNode(Shader &shader, Node* node, glm::mat4 parentTransform);
};

#endif // OPENGL_RENDERER_H
