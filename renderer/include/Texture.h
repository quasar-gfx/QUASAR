#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>

#include <glm/gtc/type_ptr.hpp>

#include <OpenGLObject.h>

struct TextureDataCreateParams {
    unsigned int width = 0;
    unsigned int height = 0;
    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool hasBorder = false;
    glm::vec4 borderColor = glm::vec4(1.0f);
    bool gammaCorrected = false;
    GLint alignment = 4;
    bool multiSampled = false;
    unsigned char* data = nullptr;
};

struct TextureFileCreateParams {
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool flipVertically = false;
    bool gammaCorrected = false;
    GLint alignment = 1;
    bool multiSampled = false;
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

    GLint alignment = 4;

    bool multiSampled = false;

    Texture() {
        target = GL_TEXTURE_2D;
    }
    Texture(const TextureDataCreateParams &params)
            : width(params.width)
            , height(params.height)
            , internalFormat(params.internalFormat)
            , format(params.format)
            , type(params.type)
            , wrapS(params.wrapS)
            , wrapT(params.wrapT)
            , minFilter(params.minFilter)
            , magFilter(params.magFilter)
            , alignment(params.alignment)
            , multiSampled(params.multiSampled) {
        target = !multiSampled ? GL_TEXTURE_2D : GL_TEXTURE_2D_MULTISAMPLE;
        loadFromData(params.data);

        if (params.hasBorder) {
            glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(params.borderColor));
        }
    }
    Texture(const TextureFileCreateParams &params)
            : type(params.type)
            , wrapS(params.wrapS)
            , wrapT(params.wrapT)
            , minFilter(params.minFilter)
            , magFilter(params.magFilter)
            , alignment(params.alignment)
            , multiSampled(params.multiSampled) {
        target = !params.multiSampled ? GL_TEXTURE_2D : GL_TEXTURE_2D_MULTISAMPLE;
        loadFromFile(params);
    }

    ~Texture() override {
        cleanup();
    }

    virtual void bind() const override {
        bind(0);
    }

    virtual void bind(unsigned int slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, ID);
    }

    virtual void unbind() const override {
        unbind(0);
    }

    virtual void unbind(unsigned int slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    void resize(unsigned int width, unsigned int height);

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    void saveAsPNG(const std::string &filename);
    void saveAsJPG(const std::string &filename, int quality = 100);
    void saveAsHDR(const std::string &filename);
#ifdef GL_CORE
    void saveDepthToFile(const std::string &filename);
#endif

protected:
    GLenum target;

    void loadFromData(unsigned char* data);
    void loadFromFile(const TextureFileCreateParams &params);
};

#endif // TEXTURE_H
