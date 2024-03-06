#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>

#include <Shader.h>
#include <Texture.h>
#include <RenderBuffer.h>
#include <Camera.h>

#define NUM_CUBEMAP_FACES 6

enum CubeMapType {
    CUBE_MAP_STANDARD,
    CUBE_MAP_SHADOW,
    CUBE_MAP_HDR,
    CUBE_MAP_PREFILTER
};

class CubeMap : public Texture {
public:
    std::vector<std::string> faceFilePaths;

    CubeMapType cubeType;

    unsigned int maxMipLevels = 1;

    CubeMap(unsigned int width, unsigned int height, CubeMapType cubeType);

    CubeMap(const std::vector<std::string> faceFilePaths,
            CubeMapType cubeType = CUBE_MAP_STANDARD,
            GLenum format = GL_RGB,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE, GLint wrapR = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR);

    ~CubeMap() {
        cleanup();
    }

    void loadFromEquirectTexture(Shader &equirectToCubeMapShader, unsigned int width, unsigned int height, Texture &equirectTexture);
    void convolve(Shader &convolutionShader, unsigned int width, unsigned int height, CubeMap &envCubeMap);
    void prefilter(Shader &prefilterShader, unsigned int width, unsigned int height, CubeMap &envCubeMap, RenderBuffer &captureRBO);

    void draw(Shader &shader, Camera* camera);

    void bind() {
        bind(0);
    }

    void bind(unsigned int slot) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_CUBE_MAP, ID);
    }

    void unbind() {
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    static const glm::mat4 captureProjection;
    static const glm::mat4 captureViews[3*NUM_CUBEMAP_FACES];

private:
    struct CubeMapVertex {
        glm::vec3 position;
    };

    GLuint quadVAO, quadVBO;

    void initBuffers();

    void loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter);

    void drawCube();
};

#endif // CUBE_MAP_H
