#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>

#include <Shader.h>
#include <Texture.h>
#include <Camera.h>

enum CubeMapType {
    CUBE_MAP_SHADOW,
    CUBE_MAP_HDR,
    CUBE_MAP_PREFILTER
};

class CubeMap : public Texture {
public:
    std::vector<std::string> faceFilePaths;

    unsigned int maxMipLevels;

    CubeMap(unsigned int width, unsigned int height, CubeMapType cubeType);

    CubeMap(std::vector<std::string> faceFilePaths,
            GLenum format = GL_RGB,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE, GLint wrapR = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR);

    ~CubeMap() {
        cleanup();
    }

    void draw(Shader &shader, Camera* camera) {
        glDepthFunc(GL_LEQUAL);

        shader.bind();

        glm::mat4 view = glm::mat4(glm::mat3(camera->getViewMatrix()));
        shader.setMat4("view", view);
        shader.setMat4("projection", camera->getProjectionMatrix());

        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_CUBE_MAP, ID);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(0);

        shader.unbind();

        // restore depth func
        glDepthFunc(GL_LESS);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    static const glm::mat4 captureProjection;
    static const glm::mat4 captureViews[18];

private:
    struct CubeMapVertex {
        glm::vec3 position;
    };

    GLuint quadVAO, quadVBO;

    void loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter);
};

#endif // CUBE_MAP_H
