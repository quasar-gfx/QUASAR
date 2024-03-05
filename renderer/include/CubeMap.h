#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>

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

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);

        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);

        glBufferData(GL_ARRAY_BUFFER, skyboxVertices.size() * sizeof(CubeMapVertex), skyboxVertices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CubeMapVertex), (void*)0);

        loadFromFiles(faceFilePaths, format, wrapS, wrapT, wrapR, minFilter, magFilter);
    }

    ~CubeMap() {
        cleanup();
    }

    GLuint quadVAO, quadVBO;

    void loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter);
};

#endif // CUBE_MAP_H
