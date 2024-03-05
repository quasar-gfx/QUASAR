#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <glad/glad.h>

#include <Shader.h>
#include <Scene.h>
#include <Camera.h>

class OpenGLRenderer {
public:
    void init();
    void drawSkyBox(Shader &shader, Scene* scene, Camera* camera);
    void draw(Shader &shader, Scene* scene, Camera* camera);

private:
    void drawNode(Shader &shader, Node* node, glm::mat4 parentTransform);
};

#endif // OPENGL_RENDERER_H
