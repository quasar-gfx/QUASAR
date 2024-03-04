#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>

#include <VertexBuffer.h>
#include <Shader.h>
#include <Camera.h>

class CubeMap {
public:
    GLuint ID;

    std::vector<std::string> faceFilePaths;

    void draw(Shader &shader, Camera* camera) {
        glDepthFunc(GL_LEQUAL);

        shader.bind();

        glm::mat4 view = glm::mat4(glm::mat3(camera->getViewMatrix()));
        shader.setMat4("view", view);
        shader.setMat4("projection", camera->getProjectionMatrix());

        quadVB->bind();
        glBindTexture(GL_TEXTURE_CUBE_MAP, ID);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glActiveTexture(GL_TEXTURE0);
        quadVB->unbind();

        shader.unbind();

        // restore depth func
        glDepthFunc(GL_LESS);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    static CubeMap* create(std::vector<std::string> faceFilePaths,
            GLenum format = GL_RGB,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE, GLint wrapR = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR) {
        return new CubeMap(faceFilePaths, format, wrapS, wrapT, wrapR, minFilter, magFilter);
    }

private:
    struct CubeMapVertex {
        glm::vec3 position;
    };

    CubeMap(std::vector<std::string> faceFilePaths,
            GLenum format = GL_RGB,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE, GLint wrapR = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR) {
        std::vector<CubeMapVertex> skyboxVertices = {
            { {-1.0f,  1.0f, -1.0f} },
            { {-1.0f, -1.0f, -1.0f} },
            { { 1.0f, -1.0f, -1.0f} },
            { { 1.0f, -1.0f, -1.0f} },
            { { 1.0f,  1.0f, -1.0f} },
            { {-1.0f,  1.0f, -1.0f} },

            { {-1.0f, -1.0f,  1.0f} },
            { {-1.0f, -1.0f, -1.0f} },
            { {-1.0f,  1.0f, -1.0f} },
            { {-1.0f,  1.0f, -1.0f} },
            { {-1.0f,  1.0f,  1.0f} },
            { {-1.0f, -1.0f,  1.0f} },

            { { 1.0f, -1.0f, -1.0f} },
            { { 1.0f, -1.0f,  1.0f} },
            { { 1.0f,  1.0f,  1.0f} },
            { { 1.0f,  1.0f,  1.0f} },
            { { 1.0f,  1.0f, -1.0f} },
            { { 1.0f, -1.0f, -1.0f} },

            { {-1.0f, -1.0f,  1.0f} },
            { {-1.0f,  1.0f,  1.0f} },
            { { 1.0f,  1.0f,  1.0f} },
            { { 1.0f,  1.0f,  1.0f} },
            { { 1.0f, -1.0f,  1.0f} },
            { {-1.0f, -1.0f,  1.0f} },

            { {-1.0f,  1.0f, -1.0f} },
            { { 1.0f,  1.0f, -1.0f} },
            { { 1.0f,  1.0f,  1.0f} },
            { { 1.0f,  1.0f,  1.0f} },
            { {-1.0f,  1.0f,  1.0f} },
            { {-1.0f,  1.0f, -1.0f} },

            { {-1.0f, -1.0f, -1.0f} },
            { {-1.0f, -1.0f,  1.0f} },
            { { 1.0f, -1.0f, -1.0f} },
            { { 1.0f, -1.0f, -1.0f} },
            { {-1.0f, -1.0f,  1.0f} },
            { { 1.0f, -1.0f,  1.0f} }
        };

        quadVB = new VertexBuffer(skyboxVertices.data(), skyboxVertices.size() * sizeof(CubeMapVertex));
        quadVB->bind();
        quadVB->addAttribute(0, 3, GL_FALSE, sizeof(CubeMapVertex), 0);

        loadFromFiles(faceFilePaths, format, wrapS, wrapT, wrapR, minFilter, magFilter);
    }

    ~CubeMap() {
        cleanup();
    }

    VertexBuffer* quadVB;

    void loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter);
};

#endif // CUBE_MAP_H
