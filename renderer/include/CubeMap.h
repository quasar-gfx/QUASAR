#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>

#include <Buffer.h>
#include <Shaders/Shader.h>
#include <Renderbuffer.h>
#include <Cameras/Camera.h>

namespace quasar {

#define NUM_CUBEMAP_FACES 6

enum class CubeMapType {
    STANDARD,
    SHADOW,
    HDR,
    PREFILTER
};

struct CubeMapCreateParams {
    uint width, height;
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

class CubeMap : public Texture {
private:
    struct CubeMapVertex {
        glm::vec3 position;
    };

public:
    Buffer vertexBuffer;

    CubeMapType type;

    uint maxMipLevels = 1;

    GLint wrapR = GL_CLAMP_TO_EDGE;

    CubeMap() : vertexBuffer(GL_ARRAY_BUFFER, sizeof(CubeMapVertex)) {
        target = GL_TEXTURE_CUBE_MAP;
    }
    CubeMap(const CubeMapCreateParams& params)
            : type(params.type)
            , wrapR(params.wrapR)
            , Texture({
                .width = params.width,
                .height = params.height,
                .format = params.format,
                .wrapS = params.wrapS,
                .wrapT = params.wrapT,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter
            })
            , vertexBuffer(GL_ARRAY_BUFFER, sizeof(CubeMapVertex)) {
        target = GL_TEXTURE_CUBE_MAP;
        if (params.rightFaceTexturePath != "" && params.leftFaceTexturePath != "" &&
            params.topFaceTexturePath != "" && params.bottomFaceTexturePath != "" &&
            params.frontFaceTexturePath != "" && params.backFaceTexturePath != "") {
            initBuffers();
            std::vector<std::string> faceTexturePaths = {
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
    ~CubeMap() override {
        glDeleteBuffers(1, &vertexArrayBuffer);
        glDeleteTextures(1, &ID);
    }

    void init(uint width, uint height, CubeMapType type);

    void loadFromEquirectTexture(const Shader& equirectToCubeMapShader, const Texture& equirectTexture) const;
    void convolve(const Shader& convolutionShader, const CubeMap& envCubeMap) const;
    void prefilter(const Shader& prefilterShader, const CubeMap& envCubeMap, Renderbuffer& captureRBO) const;

    RenderStats draw(const Shader& shader, const Camera& camera) const;

    void bind() const override {
        bind(0);
    }

    void bind(uint slot) const override {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, ID);
    }

    void unbind() const override {
        glBindTexture(target, 0);
    }

    static const glm::mat4 captureProjection;
    static const glm::mat4 captureViews[3*NUM_CUBEMAP_FACES];

private:
    GLuint vertexArrayBuffer;

    void initBuffers();
    void loadFromFiles(std::vector<std::string> faceFilePaths,
                       GLenum format,
                       GLint wrapS, GLint wrapT, GLint wrapR,
                       GLint minFilter, GLint magFilter);

    RenderStats drawCube() const;
};

} // namespace quasar

#endif // CUBE_MAP_H
