#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <string>

#include <OpenGLObject.h>

typedef GLuint TextureID;

struct TextureCreateParams {
    unsigned int width = 0;
    unsigned int height = 0;
    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool flipVertically = false;
    bool hasBorder = false;
    bool gammaCorrected = false;
    bool multiSampled = false;
    unsigned char* data = nullptr;
    std::string path = "";
};

class Texture : public OpenGLObject {
public:
    unsigned int width, height;

    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;

    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;

    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;

    bool multiSampled = false;

    explicit Texture() = default;
    explicit Texture(const TextureCreateParams &params)
            : width(params.width)
            , height(params.height)
            , internalFormat(params.internalFormat)
            , format(params.format)
            , type(params.type)
            , wrapS(params.wrapS)
            , wrapT(params.wrapT)
            , minFilter(params.minFilter)
            , magFilter(params.magFilter)
            , multiSampled(params.multiSampled) {
        target = !multiSampled ? GL_TEXTURE_2D : GL_TEXTURE_2D_MULTISAMPLE;
        if (params.path == "") {
            init(params);
        }
        else {
            loadFromFile(params);
        }
    }

    void bind() {
        bind(0);
    }

    void bind(unsigned int slot = 0) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, ID);
    }

    void unbind() {
        unbind(0);
    }

    void unbind(unsigned int slot) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    void resize(unsigned int width, unsigned int height);

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    void saveTextureToPNG(std::string filename);
    void saveTextureToJPG(std::string filename, int quality = 100);
    void saveTextureToHDR(std::string filename);
    void saveDepthToFile(std::string filename);

private:
    GLenum target;

    void init(const TextureCreateParams &params);
    void loadFromFile(const TextureCreateParams &params);
};

#endif // TEXTURE_H
