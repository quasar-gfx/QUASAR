#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Renderbuffer.h>
#include <Camera.h>

#define NUM_CUBEMAP_FACES 6

enum class CubeMapType {
    STANDARD,
    SHADOW,
    HDR,
    PREFILTER
};

struct CubeMapCreateParams {
    unsigned int width, height;
    std::string rightFaceTexturePath = "";
    std::string leftFaceTexturePath = "";
    std::string topFaceTexturePath = "";
    std::string bottomFaceTexturePath = "";
    std::string frontFaceTexturePath = "";
    std::string backFaceTexturePath = "";
    CubeMapType type = CubeMapType::STANDARD;
    GLenum format = GL_RGB;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint wrapR = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
};

class CubeMap : public OpenGLObject {
public:
    CubeMapType type;

    unsigned int maxMipLevels = 1;

    unsigned int width, height;

    explicit CubeMap() = default;

    explicit CubeMap(const CubeMapCreateParams &params) : type(params.type), width(params.width), height(params.height) {
        if (params.rightFaceTexturePath != "" && params.leftFaceTexturePath != "" &&
            params.topFaceTexturePath != "" && params.bottomFaceTexturePath != "" &&
            params.frontFaceTexturePath != "" && params.backFaceTexturePath != "") {
            initBuffers();
            std::vector faceTexturePaths = {
                params.rightFaceTexturePath,
                params.leftFaceTexturePath,
                params.topFaceTexturePath,
                params.bottomFaceTexturePath,
                params.frontFaceTexturePath,
                params.backFaceTexturePath
            };
            loadFromFiles(faceTexturePaths, params.format, params.wrapS, params.wrapT, params.wrapR, params.minFilter, params.magFilter);
        }
        else {
            init(params.width, params.height, params.type);
        }
    }

    ~CubeMap() {
        cleanup();
    }

    void init(unsigned int width, unsigned int height, CubeMapType type);

    void loadFromEquirectTexture(Shader &equirectToCubeMapShader, Texture &equirectTexture);
    void convolve(Shader &convolutionShader, CubeMap &envCubeMap);
    void prefilter(Shader &prefilterShader, CubeMap &envCubeMap, Renderbuffer &captureRBO);

    void draw(Shader &shader, Camera &camera);

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
